"""Build the evaluation and the deployment fit from one immutable set of source snapshots."""

import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import sqlalchemy as sa

from . import storage as db
from .calendar import annual_periods
from .charts import draw
from .features import build_features, make_targets
from .ingest import save_json
from .model import (
    backtest,
    cost_sensitivity,
    evaluate,
    fit_model,
    forecast,
    forecast_metrics,
    incremental,
    select_model,
)
from .report import write_dashboard, write_evaluation, write_outlook

LOG = logging.getLogger(__name__)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def records(frame):
    return json.loads(
        frame.to_json(orient="records", date_format="iso", date_unit="us", double_precision=15)
    )


def run(store, out, cfg, *, experiment=None, datasets=None):
    """Publish a run after all artifacts are ready; retrying failed stages is safe."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory to preserve prior results")
    out.parent.mkdir(parents=True, exist_ok=True)
    store.bind(cfg)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        manifest = build(store, stage, cfg, experiment, datasets)
        if out.exists():
            out.rmdir()
        stage.rename(out)
    LOG.info("Run completed mode=%s path=%s", manifest["mode"], out)
    return manifest


def build(store, out, cfg, experiment, datasets):
    prices, posts, news = [store.frame(t) for t in ["prices", "posts", "news"]]
    if prices.empty or posts.empty:
        raise ValueError("Both prices and posts are required before evaluation")
    prices = prices.sort_values("time").copy()
    posts = posts[posts.platform == cfg.platform].copy()
    if datasets:
        posts = posts.loc[posts.source.isin(datasets)]
    meta = store.metadata()
    synthetic = prices.source.eq("SYNTHETIC_DEMO").all()
    if (prices.source.eq("SYNTHETIC_DEMO").any() or posts.source.eq("SYNTHETIC_DEMO").any()) and not (
        synthetic and posts.source.eq("SYNTHETIC_DEMO").all()
    ):
        raise ValueError("Mixed synthetic and real data")
    if len(prices.source.unique()) != 1:
        raise ValueError("Use a single consistently adjusted price series per experiment")
    exp = experiment or (
        "demo"
        if synthetic
        else "D"
        if cfg.platform == "StockTwits"
        else "A"
        if posts.source.eq("apify").all()
        else "B"
    )
    if (exp == "D") != (cfg.platform == "StockTwits") or (exp == "demo") != synthetic:
        raise ValueError("Experiment, platform and source mode disagree")
    if exp == "A":
        posts = posts.loc[posts.source == "apify"]
        end = pd.Timestamp(prices.time.iloc[-1]) if not cfg.end else pd.Timestamp(cfg.end)
        dates = pd.to_datetime(prices.time, utc=True, format="mixed")
        end = pd.to_datetime(end, utc=True)
        prices = prices.loc[(dates > end - pd.Timedelta(days=cfg.days)) & (dates <= end)]
    elif exp == "B":
        posts = posts.loc[posts.source != "apify"]
    elif exp == "C":
        if set(posts.source.eq("apify").unique()) != {True, False}:
            raise ValueError("Combined experiment requires both historical X and recent Apify X")
        ranges = (
            pd.to_datetime(posts.created_at, utc=True, format="mixed")
            .groupby(posts.source.eq("apify"))
            .agg(["min", "max"])
        )
        if ranges["min"].max() > ranges["max"].min() + pd.Timedelta(days=1):
            raise ValueError("The two source windows are disjoint; evaluate A and B separately")
    if cfg.start:
        prices = prices.loc[
            pd.to_datetime(prices.time, utc=True, format="mixed") > pd.to_datetime(cfg.start, utc=True)
        ]
    if cfg.end:
        prices = prices.loc[
            pd.to_datetime(prices.time, utc=True, format="mixed") <= pd.to_datetime(cfg.end, utc=True)
        ]
    mode = "SYNTHETIC_DEMO" if synthetic else "REAL_HISTORICAL" if exp in ("B", "D") else "REAL_DATA"
    frame, pc, ac, audit, scored = build_features(prices, posts, cfg, with_posts=True)
    audit["price_providers"] = sorted(prices.source.unique().tolist())
    with store.engine.connect() as con:
        audit["raw_payloads_in_sources"] = con.scalar(
            sa.select(sa.func.count())
            .select_from(db.raw)
            .where(db.raw.c.source.in_(posts.source.unique().tolist()), db.raw.c.platform == cfg.platform)
        )
    audit["normalized_source_rows"] = len(posts)
    for key in ("historical_sample", "historical_import", "stocktwits_sample"):
        if key in meta:
            audit[key] = meta[key]
    summaries = scored.groupby("source").agg(
        posts=("id", "size"),
        mean=("sentiment", "mean"),
        median=("sentiment", "median"),
        dispersion=("sentiment", "std"),
        first=("created_at", "min"),
        last=("created_at", "max"),
    )
    summaries.to_csv(out / "source_distributions.csv")
    if exp == "C" and len(summaries) < 2:
        raise ValueError("Both source categories need usable posts in the price window")
    frame.to_csv(out / "features.csv")
    pred, folds = evaluate(frame, pc, ac, cfg)
    selected, selection_scores = select_model(pred)
    metrics, fold_metrics = forecast_metrics(pred), forecast_metrics(pred, True)
    gains = incremental(pred, fold_metrics, cfg.seed)
    periods = annual_periods(frame.index[0], cfg)
    risk, curves = backtest(pred, cfg, periods)
    sensitivity = cost_sensitivity(pred, cfg, periods)
    cols = pc if selected in ("persistence", "price_only") else ac
    model, projections = forecast(frame, cols, cfg, selected)
    fit_time = db.timestamp()
    cutoff = frame.index[-1].isoformat()
    versions = {}
    for pkg in (
        "numpy",
        "pandas",
        "scikit-learn",
        "vaderSentiment",
        "plotly",
        "SQLAlchemy",
        "matplotlib",
        "exchange-calendars",
        "yfinance",
        "apify-client",
        "tavily-python",
        "exa-py",
        "torch",
        "transformers",
    ):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    inputs = {}
    for name, df, order in [("prices", prices, "time"), ("posts", posts, "id"), ("news", news, "url")]:
        path = out / f"input_{name}.csv"
        df.sort_values(order, kind="stable").to_csv(path, index=False)
        inputs[name] = dict(rows=len(df), sha256=digest(path))
    source_hash = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    spec = dict(
        experiment=exp,
        config=cfg.to_dict(),
        inputs=inputs,
        source_sha256=source_hash,
        engine=audit["engine"],
        libraries=versions,
    )
    run_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    prediction_info = dict(
        mode=mode,
        asset=cfg.asset,
        selected_model=selected,
        model_version=run_id,
        training_timestamp=fit_time,
        data_cutoff=cutoff,
        feature_cutoff=cutoff,
        prediction_timestamp=fit_time,
        forecast_origin=cutoff,
        forecasts=projections,
    )
    save_json(out / "forecast.json", prediction_info)
    pd.DataFrame(projections).to_csv(out / "forecast.csv", index=False)
    matrix = (
        pred[pred.model == selected]
        .pivot(index="time", columns="horizon", values="predicted_price")
        .add_prefix("predicted_n+")
    )
    matrix.insert(0, "observed_n+0", frame.close.reindex(matrix.index))
    matrix.loc[frame.index[-1]] = [p["price"] for p in projections]
    matrix.to_csv(out / "forecast_horizon_matrix.csv")
    for filename, data in [
        ("predictions", pred),
        ("forecast_metrics", metrics),
        ("fold_metrics", fold_metrics),
        ("incremental", gains),
        ("strategy_metrics", risk),
        ("strategy_returns", curves),
        ("strategy_cost_sensitivity", sensitivity),
    ]:
        data.to_csv(out / (filename + ".csv"), index=False)
    curves[(curves.position > 0) & (curves.model != "buy_hold")].to_csv(out / "trades.csv", index=False)
    save_json(out / "data_quality.json", audit)
    save_json(out / "folds.json", folds)
    joblib.dump(
        dict(
            model=model,
            features=cols,
            config=cfg.to_dict(),
            mode=mode,
            trained_asof=cutoff,
            selected_model=selected,
            version=run_id,
        ),
        out / "model.joblib",
    )
    y = make_targets(frame).dropna()
    ablation = (
        model if selected == "sentiment_price" else fit_model(frame.loc[y.index, ac], y, cfg.ridge_alpha)
    )
    coefficients = pd.Series(ablation[-1].coef_[0], index=ablation[0].get_feature_names_out(ac))
    draw(out, frame, pred, curves, metrics, fold_metrics, selected, cfg, coefficients)
    write_evaluation(
        out, metrics, risk, folds, cfg, audit, mode, gains, selected, selection_scores, meta, periods
    )
    err = metrics.query("phase=='holdout' and model==@selected and horizon==1").mae_usd.iloc[0]
    write_outlook(out, frame, news, projections, mode, audit, cfg, selected, err)
    write_dashboard(out, frame, pred, curves, projections, mode, cfg)
    manifest = dict(
        **spec,
        id=run_id,
        mode=mode,
        created_at=fit_time,
        python=platform.python_version(),
        features=ac,
        price_only_features=pc,
        selected_model=selected,
        selection_log_rmse=selection_scores,
        periods_per_year=periods,
        date_range=[audit["first_close_utc"], cutoff],
        training_last_origin=y.index[-1].isoformat(),
        feature_cutoff=cutoff,
        source_datasets=summaries.index.tolist(),
        collection_metadata=meta,
        apify=meta.get("x_collection"),
        news_providers=sorted(news.source.unique().tolist()),
        price_providers=audit["price_providers"],
        models={
            "persistence": "zero cumulative log return",
            "ridge_alpha": cfg.ridge_alpha,
            "boosting": dict(
                max_iter=45, max_leaf_nodes=7, learning_rate=0.04, min_samples_leaf=35, l2_regularization=10
            ),
        },
        artifact_sha256={p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file()},
    )
    save_json(out / "manifest.json", manifest)
    with store.engine.begin() as con:
        enriched = scored.assign(engine=audit["engine"], experiment=run_id, processed_at=fit_time)
        store.upsert(db.processed, records(enriched[list(db.processed.columns.keys())]), con)
        for table, columns in [(db.features, ac), (db.sentiment, [c for c in ac if c not in pc])]:
            payload = records(frame[columns])
            store.upsert(
                table,
                [
                    dict(experiment=run_id, time=t.isoformat(), values=v)
                    for t, v in zip(frame.index, payload)
                ],
                con,
            )
        store.upsert(db.forecasts, records(pred.assign(experiment=run_id)), con)
        store.upsert(
            db.forecasts,
            [
                dict(
                    experiment=run_id,
                    model=selected,
                    time=cutoff,
                    horizon=p["horizon"],
                    phase="deployment",
                    target_time=p["time"],
                    origin_price=projections[0]["price"],
                    predicted_price=p["price"],
                    predicted_return=p["log_return"],
                )
                for p in projections
            ],
            con,
        )
        store.upsert(
            db.metrics,
            [
                dict(experiment=run_id, phase=r["phase"], model=r["model"], horizon=r["horizon"], values=r)
                for r in records(metrics)
            ],
            con,
        )
        store.upsert(db.experiments, [dict(id=run_id, manifest=manifest)], con)
    return manifest
