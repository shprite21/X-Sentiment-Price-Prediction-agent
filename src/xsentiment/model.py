"""Direct multi-horizon models, strictly separated development and final test periods."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .calendar import future_closes
from .features import make_targets


def fit_model(x, y, alpha, name="sentiment_price", seed=42):
    if name == "persistence":
        model = DummyRegressor(strategy="constant", constant=np.zeros(y.shape[1]))
    elif name == "boosting":
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=45,
                max_leaf_nodes=7,
                learning_rate=0.04,
                min_samples_leaf=35,
                l2_regularization=10,
                early_stopping=False,
                random_state=seed,
            )
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
    with threadpool_limits(limits=2):
        model.fit(x, y)
    return model


def splits(n, cfg):
    """Purge five bar origins both before each fit/test boundary and before final selection."""
    holdout = cfg.holdout or max(1, int(0.17 * n))
    first_holdout = n - holdout
    gap = cfg.horizons + cfg.embargo
    development_end = first_holdout - gap
    first_test = (cfg.min_train or max(30, int(0.60 * development_end))) + gap
    if first_test >= development_end or holdout < 2:
        raise ValueError(f"Insufficient labeled origins ({n}) for train, purge, walk-forward and holdout")
    for start in range(first_test, development_end, cfg.test_size):
        yield (
            "walk_forward",
            np.arange(start - gap),
            np.arange(start, min(start + cfg.test_size, development_end)),
        )
    yield "holdout", np.arange(first_holdout - gap), np.arange(first_holdout, n)


def evaluate(
    frame,
    price_cols,
    all_cols,
    cfg,
    names=("persistence", "price_only", "sentiment_price", "boosting"),
    *,
    earlier_frame=None,
):
    targets = make_targets(frame, cfg.horizons).dropna()
    if earlier_frame is not None:
        if not earlier_frame.index.is_monotonic_increasing or earlier_frame.index[-1] >= frame.index[0]:
            raise ValueError("The archival segment must end before the recent segment begins")
        old_y = make_targets(earlier_frame, cfg.horizons).dropna()
        old_x = earlier_frame.loc[old_y.index]
    records, folds = [], []
    for fold, (phase, tr, te) in enumerate(splits(len(targets), cfg), 1):
        train, test = frame.iloc[tr], frame.iloc[te]
        y_train = targets.iloc[tr]
        if earlier_frame is not None:
            train = pd.concat([old_x, train])
            y_train = pd.concat([old_y, y_train])
        last_label = frame.index[tr[-1] + cfg.horizons]
        if last_label >= test.index[0]:
            raise ValueError("Training labels cross the forecast origin")
        # Fixed rule: training median absolute one-bar log return, with a round-trip cost floor.
        threshold = max(float(y_train[1].abs().median()), 2 * cfg.cost_bps / 10000)
        folds.append(
            dict(
                fold=fold,
                phase=phase,
                train_rows=len(train),
                recent_train_rows=len(tr),
                archive_train_rows=len(train) - len(tr),
                test_rows=len(te),
                train_start=train.index[0].isoformat(),
                train_end=train.index[-1].isoformat(),
                last_train_label=last_label.isoformat(),
                test_start=test.index[0].isoformat(),
                test_end=test.index[-1].isoformat(),
                last_test_label=frame.index[te[-1] + 5].isoformat(),
                threshold=threshold,
            )
        )
        for name in names:
            cols = price_cols if name in {"price_only", "persistence"} else all_cols
            model = fit_model(train[cols], y_train, cfg.ridge_alpha, name, cfg.seed)
            with threadpool_limits(limits=2):
                yhat = model.predict(test[cols])
            for h in range(1, cfg.horizons + 1):
                actual = frame.iloc[te + h]
                records.append(
                    pd.DataFrame(
                        dict(
                            time=test.index,
                            target_time=actual.index,
                            phase=phase,
                            fold=fold,
                            model=name,
                            horizon=h,
                            origin_price=test.close.to_numpy(),
                            predicted_return=yhat[:, h - 1],
                            actual_return=targets.iloc[te][h].to_numpy(),
                            predicted_price=test.close.to_numpy() * np.exp(yhat[:, h - 1]),
                            actual_price=actual.close.to_numpy(),
                            target_open=actual.open.to_numpy(),
                            threshold=threshold,
                        )
                    )
                )
    return pd.concat(records, ignore_index=True), folds


def select_model(predictions):
    development = predictions.loc[predictions.phase == "walk_forward"]
    error = development.predicted_return - development.actual_return
    scores = error.pow(2).groupby(development.model).mean().pow(0.5).sort_values()
    return str(scores.index[0]), scores.to_dict()


def forecast(frame, cols, cfg, name="sentiment_price"):
    y = make_targets(frame, cfg.horizons).dropna()
    model = fit_model(frame.loc[y.index, cols], y, cfg.ridge_alpha, name, cfg.seed)
    with threadpool_limits(limits=2):
        yhat = model.predict(frame[cols].iloc[[-1]])[0]
    stamp, price = frame.index[-1], float(frame.close.iloc[-1])
    result = [
        dict(
            horizon=0,
            time=stamp.isoformat(),
            price=price,
            kind="observed",
            log_return=0.0,
            simple_return=0.0,
        )
    ]
    for h, (end, value) in enumerate(zip(future_closes(stamp, cfg), yhat), 1):
        result.append(
            dict(
                horizon=h,
                time=end.isoformat(),
                price=float(price * np.exp(value)),
                kind="forecast",
                log_return=float(value),
                simple_return=float(np.expm1(value)),
            )
        )
    return model, result


def forecast_metrics(predictions, by_fold=False):
    rows = []
    keys = ["phase", "model", "horizon"] + (["fold"] if by_fold else [])
    for idx, g in predictions.groupby(keys):
        err = g.predicted_price - g.actual_price
        re = g.predicted_return - g.actual_return
        baseline = (g.origin_price - g.actual_price).abs().mean()

        def r2(series, residual):
            denom = ((series - series.mean()) ** 2).sum()
            return 1 - (residual**2).sum() / denom if denom > 0 else np.nan

        rows.append(
            {
                **dict(zip(keys, idx)),
                "n": len(g),
                "mae_usd": err.abs().mean(),
                "rmse_usd": np.sqrt((err**2).mean()),
                "r2_price": r2(g.actual_price, err),
                "r2_return": r2(g.actual_return, re),
                "mean_error_usd": err.mean(),
                "direction_accuracy": (np.sign(g.predicted_return) == np.sign(g.actual_return)).mean(),
                "mae_skill_vs_persistence": 1 - err.abs().mean() / baseline if baseline > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def incremental(predictions, fold_metrics, seed):
    """Paired circular moving-block bootstrap of the reduction in dollar absolute error."""
    rng = np.random.default_rng(seed)
    rows = []
    for (phase, h), g in predictions.groupby(["phase", "horizon"]):
        price = g[g.model == "price_only"].set_index("time").sort_index()
        both = g[g.model == "sentiment_price"].set_index("time").reindex(price.index)
        a = price.predicted_price - price.actual_price
        b = both.predicted_price - both.actual_price
        difference = (a.abs() - b.abs()).to_numpy()
        n = len(a)
        block = min(n, max(h, int(np.ceil(n ** (1 / 3)))))
        starts = rng.integers(n, size=(500, int(np.ceil(n / block))))
        sample = (starts[:, :, None] + np.arange(block)) % n
        interval = np.quantile(difference[sample.reshape(500, -1)[:, :n]].mean(axis=1), [0.025, 0.975])
        f = fold_metrics.query("phase == @phase and horizon == @h").pivot(
            index="fold", columns="model", values="mae_usd"
        )
        direction = lambda df: (np.sign(df.predicted_return) == np.sign(df.actual_return)).mean()
        rows.append(
            dict(
                phase=phase,
                horizon=h,
                rmse_gain_usd=np.sqrt((a * a).mean()) - np.sqrt((b * b).mean()),
                mae_gain_usd=difference.mean(),
                direction_gain_pp=100 * (direction(both) - direction(price)),
                mae_ci_low=interval[0],
                mae_ci_high=interval[1],
                block_bars=block,
                winning_folds=int((f.sentiment_price < f.price_only).sum()),
                folds=len(f),
            )
        )
    return pd.DataFrame(rows)


def strategy_returns(group, cost_bps, benchmark=False):
    """Forecast at close t. Long/cash next open to next close, liquidating every bar."""
    one = group.loc[group.horizon == 1].sort_values("time")
    position = (one.predicted_return > one.threshold).to_numpy(dtype=float)
    realized = one.actual_price.to_numpy() / one.target_open.to_numpy() - 1
    turnover = 2 * position
    if benchmark:
        position[:] = 1
        realized[1:] = one.actual_price.to_numpy()[1:] / one.actual_price.to_numpy()[:-1] - 1
        turnover[:] = 0
        turnover[0] += 1
        turnover[-1] += 1
    gross = position * realized
    rate = cost_bps / 10000
    # Multiplicative fee/slippage on each executed leg.
    net = (1 + gross) * (1 - rate) ** turnover - 1
    curve = pd.DataFrame(
        dict(
            time=one.target_time.to_numpy(),
            signal_time=one.time.to_numpy(),
            position=position,
            gross_return=gross,
            turnover=turnover,
            net_return=net,
            cost=gross - net,
            fold=one.fold.to_numpy(),
        )
    )
    curve["equity"] = (1 + curve.net_return).cumprod()
    curve["drawdown"] = curve.equity / curve.equity.cummax().clip(lower=1) - 1
    return curve


def risk_metrics(returns, periods=8760):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid strategy returns")
    mean, std = r.mean(), r.std(ddof=1) if len(r) > 1 else 0
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    equity = np.r_[1.0, np.cumprod(1 + r)]
    return dict(
        sharpe=float(mean / std * np.sqrt(periods)) if std > 0 else None,
        sortino=float(mean / downside * np.sqrt(periods)) if downside > 0 else None,
        max_drawdown=float((equity / np.maximum.accumulate(equity) - 1).min()),
        total_return=float(equity[-1] - 1),
        annualized_return=(
            float(np.expm1(np.log1p(r).mean() * periods)) if np.log1p(r).mean() * periods < 709 else None
        ),
        bars=len(r),
    )


def backtest(predictions, cfg, periods=8760):
    curves, rows = [], []
    for (phase, name), group in predictions.groupby(["phase", "model"]):
        variants = [name, "buy_hold"] if name == "persistence" else [name]
        for variant in variants:
            benchmark = variant == "buy_hold"
            curve = strategy_returns(group, cfg.cost_bps, benchmark)
            curve["phase"], curve["model"] = phase, variant
            curves.append(curve)
            for basis in ("gross", "net"):
                returns = curve[basis + "_return"]
                values = risk_metrics(returns, periods)
                trades = returns[curve.position > 0]
                count = 1 if benchmark else len(trades)
                wins = float(values["total_return"] > 0) if benchmark else (trades > 0).mean()
                rows.append(
                    dict(
                        phase=phase,
                        model=variant,
                        basis=basis,
                        **values,
                        trades=count,
                        win_rate=wins,
                        turnover=curve.turnover.sum(),
                        exposure=curve.position.mean(),
                    )
                )
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def cost_sensitivity(predictions, cfg, periods=8760):
    rows = []
    for (phase, name), group in predictions.groupby(["phase", "model"]):
        for bps in (0.0, 5.0, 10.0, 25.0):
            curve = strategy_returns(group, bps)
            rows.append(
                dict(phase=phase, model=name, cost_bps=bps, **risk_metrics(curve.net_return, periods))
            )
    return pd.DataFrame(rows)
