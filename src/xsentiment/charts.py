"""Static research figures, written without network access or a display server."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def draw(out, frame, pred, curves, metrics, fold_metrics, chosen, cfg, coefficients):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dest = out / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.axisbelow": True,
        }
    )
    names = {
        "price_only": "Price Ridge",
        "sentiment_price": "Sentiment + price Ridge",
        "persistence": "Random walk",
        "boosting": "Gradient boosting",
        "buy_hold": "Buy and hold",
    }

    def save(fig, filename):
        fig.savefig(dest / (filename + ".png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def dates(ax):
        loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5), layout="constrained")
    axes[0].plot(frame.index, frame.close, lw=1, color="#184c69")
    axes[0].set(
        ylabel="Close (quote currency)",
        title=f"{cfg.asset}  |  Price and {cfg.platform} sentiment at the same close",
    )
    axes[1].scatter(frame.index, frame.sentiment, s=3, alpha=0.5, color="#a74723")
    axes[1].set(ylabel="Mean score", xlabel="Bar close (UTC)", ylim=(-1.05, 1.05))
    dates(axes[1])
    save(fig, "sentiment_price")
    day = frame.resample("1D")
    daily = pd.DataFrame(
        {
            "Mean score": day.apply(
                lambda z: (
                    (z.sentiment * z.post_count).sum() / z.post_count.sum()
                    if z.post_count.sum()
                    else np.nan
                )
            ),
            "Posts": day.post_count.sum(),
        }
    )
    for label, col in [("Positive share", "bull_fraction"), ("Negative share", "bear_fraction")]:
        daily[label] = (frame[col] * frame.post_count).resample("1D").sum() / daily.Posts.replace(0, np.nan)
    fig, axes = plt.subplots(4, 1, figsize=(10, 4.7), sharex=True, layout="constrained")
    for ax, (label, values), palette, bounds in zip(
        axes, daily.items(), ["RdBu", "Blues", "Greens", "Reds"], [(-1, 1), (0, None), (0, 1), (0, 1)]
    ):
        cm = plt.get_cmap(palette).with_extremes(bad="#d8d9da")
        pic = ax.imshow(
            values.to_numpy()[None, :],
            aspect="auto",
            interpolation="nearest",
            cmap=cm,
            vmin=bounds[0],
            vmax=bounds[1],
        )
        ax.set_yticks([0], [label])
        ax.grid(False)
        fig.colorbar(pic, ax=ax, shrink=0.8, pad=0.015)
    ticks = np.unique(np.linspace(0, len(daily) - 1, min(7, len(daily))).astype(int))
    axes[-1].set_xticks(ticks, daily.index[ticks].strftime("%d %b %y"))
    axes[0].set_title(f"{cfg.platform} daily aggregates  |  Grey indicates unavailable sentiment")
    save(fig, "sentiment_heatmap")
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8), layout="constrained")
    hold = pred[(pred.phase == "holdout") & (pred.model == chosen)]
    for (h, g), ax in zip(hold.groupby("horizon"), axes):
        ax.plot(g.target_time, g.actual_price, color="#184c69", label="Observed", lw=1.2)
        ax.plot(g.target_time, g.predicted_price, color="#c55828", label="Forecast", lw=0.85, alpha=0.9)
        ax.set_ylabel(f"n+{h} price")
    axes[0].legend(ncol=2)
    axes[0].set_title(f"{cfg.asset}  |  Final holdout, {names[chosen]} aligned at target close")
    dates(axes[-1])
    save(fig, "predicted_actual")
    fig, axes = plt.subplots(1, 5, figsize=(11, 3), sharey=True, layout="constrained")
    for h, ax in enumerate(axes, 1):
        y = 100 * np.log(frame.close.shift(-h) / frame.close)
        ok = frame.sentiment.notna() & y.notna()
        ax.scatter(frame.sentiment[ok], y[ok], s=5, alpha=0.22, color="#306f85")
        ax.axhline(0, color="#555", lw=0.5)
        ax.set(xlabel="Mean score", title=f"n+{h}", xlim=(-1.05, 1.05))
    axes[0].set_ylabel("Forward log return (%)")
    fig.suptitle("Observed sentiment and subsequent returns across the research window")
    save(fig, "sentiment_forward_returns")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for model, g in metrics[metrics.phase == "holdout"].groupby("model"):
        axes[0].plot(g.horizon, g.rmse_usd, marker="o", label=names[model])
    for model, g in fold_metrics.query("phase == 'walk_forward' and horizon == 1").groupby("model"):
        axes[1].plot(g.fold, g.mae_usd, marker=".", label=names[model])
    axes[0].set(title="Holdout RMSE by horizon", xlabel="Bars ahead", ylabel="Quote currency")
    axes[1].set(title="Walk-forward one-bar MAE", xlabel="Fold", ylabel="Quote currency")
    axes[0].legend(fontsize=8)
    save(fig, "model_evaluation")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5), layout="constrained")
    for model, g in curves[curves.phase == "holdout"].groupby("model"):
        axes[0].plot(g.time, g.equity, lw=1.2, label=names[model])
        axes[1].plot(g.time, 100 * g.drawdown, lw=1)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set(title="Final holdout after fees and slippage", ylabel="Equity (initial 1)")
    axes[1].set(ylabel="Drawdown (%)", xlabel="Bar close (UTC)")
    dates(axes[1])
    save(fig, "strategy_equity_drawdown")
    coef = coefficients.abs().nlargest(12).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.barh(coef.index, coefficients[coef.index], color="#306f85")
    ax.set(
        title="Sentiment + price Ridge  |  One-bar coefficients after deployment refit",
        xlabel="Change in predicted log return per standardized input unit",
    )
    save(fig, "ridge_coefficients")
