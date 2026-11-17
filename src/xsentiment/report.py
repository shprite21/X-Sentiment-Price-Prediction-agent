import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def markdown_table(frame, digits=4):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "N/A" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)

    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    rows += [
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def write_evaluation(
    out, metrics, risk, folds, cfg, audit, mode, gains, selected, scores, metadata, periods
):
    synthetic = mode == "SYNTHETIC_DEMO"
    coverage = audit["hour_coverage_fraction"]
    wf = gains[gains.phase == "walk_forward"]
    ho = gains[gains.phase == "holdout"]
    count = lambda df, col: int((df[col] > 0).sum())
    history = metadata.get("historical_sample")
    provenance = (
        (
            "The archive contains {:,} CSV records. The parser rejected {:,} malformed rows. "
            "{:,} valid originals fell inside the window. A seeded hash sample retained up to {} per UTC hour "
            "before scoring. The cap changes the interpretation of volume. Author display names are an imperfect "
            "proxy for identity. The source supplies no post IDs, language field or reliable per-post engagement. "
            "Timestamp plus exact text identifies a post."
        ).format(
            history["rows_scanned"],
            history["invalid_rows"],
            history["eligible_in_window"],
            history["sample_cap_per_hour"],
        )
        if history
        else ""
    )
    if cfg.platform == "StockTwits":
        provenance = (
            "The public S3 systematic byte-range sample contains BTC.X-tagged messages with user-supplied "
            "bullish/bearish labels. The experiment uses date precision with a two-day UTC availability delay and "
            "strict-before-close binning. Text was not present in the label files. Labelled users are self-selected. "
            "Price bars are full UTC days aggregated from the historical hourly Coinbase candles."
        )
    if mode == "REAL_DATA" and metadata.get("x_collection"):
        collection = metadata["x_collection"]
        run_count = len(collection["runs"])
        provenance = (
            f"Apify Actor {collection['actor_id']} supplied a capped sample across {run_count} runs. "
            "The manifest contains the actual query inputs, run and dataset IDs, and charge records. "
            "Coregent searches return relevance-ranked results within each date stratum. Retrospective "
            "ranking can depend on popularity accumulated after publication. Market-wide post volume and "
            "the set of posts visible to a contemporaneous collector are unknown."
        )
    confirmation = (ho.mae_ci_low > 0).sum()
    engine = audit["engine"].split(":")[0]
    summary = (
        f"Adding {cfg.platform} sentiment to Ridge reduced walk-forward MAE at {count(wf, 'mae_gain_usd')}/5 "
        f"horizons and RMSE at {count(wf, 'rmse_gain_usd')}/5. On the final holdout the counts were "
        f"{count(ho, 'mae_gain_usd')}/5 and {count(ho, 'rmse_gain_usd')}/5. At {confirmation}/5 horizons "
        "the paired holdout interval for MAE reduction lay entirely above zero."
    )
    verdict = (
        "The experiment supports a sample-specific incremental effect at those horizons."
        if confirmation
        else "The sample provides no clear evidence of a repeatable reduction in absolute forecast error."
    )
    chosen_risk = risk.query("phase == 'holdout' and model == @selected and basis == 'net'")
    name = (
        "Synthetic workflow verification"
        if synthetic
        else "Supplementary StockTwits research"
        if cfg.platform == "StockTwits"
        else "Historical X research"
        if mode == "REAL_HISTORICAL"
        else "X research"
    )
    sections = [
        f"# {cfg.asset} | {name}",
        f"**{mode.replace('_', ' ')}**. {summary} {verdict}",
        f"**Validation selected {selected}** by pooled forward log-return RMSE with equal weight per origin and horizon. "
        "Selection scores used only walk-forward labels maturing before the final holdout. "
        "The model was then frozen for the final block.",
        "## Data and availability",
        f"The {cfg.interval} sample contains {audit['price_rows']:,} completed bars "
        f"from {audit['first_close_utc']} through {audit['last_close_utc']}. {audit['input_posts']:,} source posts "
        f"yielded {audit['unique_text_posts_in_window']:,} scored observations. ID duplicates within the aligned window "
        f"totalled {audit['duplicate_ids']:,}, repeated cleaned texts {audit['duplicate_text']:,}, and empty or "
        f"off-topic texts {audit['irrelevant_or_empty']:,}. After 72 bars of warmup, {audit['usable_feature_rows']:,} "
        f"origins remained, including {audit['hours_without_posts']:,} without posts. Coverage was {coverage:.1%}. "
        f"{audit['labeled_observations']:,} origins had all five matured targets. Price sources: {', '.join(audit['price_providers'])}.",
        provenance,
        "Posts are grouped strictly before each close. A reported availability time delays a post. "
        f"The run used the {cfg.availability} policy. Publication-time reconstruction assumes archived text could have "
        "been read at publication. Retrospective deletions, edits and collection delays remain unobservable. "
        "First-seen mode requires a collector observation before the origin. Engagement, follower snapshots, "
        "exchange volume and news are excluded from both model matrices.",
        "## Sentiment and supervised models",
        f"The sentiment engine was {engine}. FinBERT uses positive minus negative "
        "softmax probability and the most likely class. The lightweight VADER option uses normalized lexical "
        "weights and a ±0.05 polarity band. StockTwits uses its own binary label. Missing intervals retain "
        "undefined sentiment, zero sampled posts and an explicit presence indicator. "
        "Returns, acceleration, trading ranges, rolling extremes, volatility and moving-average ratios use trailing "
        "prices. Sentiment aggregates include dispersion, proportions, imbalance, lags, momentum and log counts. "
        "Observed neutral posts remain in their interval.",
        f"Five direct targets are log(close[t+h]/close[t]). Multiplying the observed n+0 close by the exponential "
        f"of each predicted return gives the price forecast. Random walk, price Ridge, price + sentiment Ridge "
        f"(alpha {cfg.ridge_alpha:g}) and five regularized gradient-boosting estimators use identical origins. "
        "Ridge fits median imputation, missingness indicators and scaling separately on each training fold. "
        "Boosting handles missing values natively, with 45 iterations, at most seven leaves and fixed seed. "
        "Observable future closes provide supervised labels. Learning a reinforcement policy would require an "
        "execution and reward environment that these data do not identify.",
        "## Expanding-window evaluation",
        f"The table gives realized fold sizes. Each fit purges five origins "
        f"plus {cfg.embargo} embargo bars. The last development test label also precedes the first final holdout "
        "origin. Model settings were fixed. The final block is held out within this protocol. "
        "This repository had previously evaluated an earlier specification on the historical period, which limits "
        "its status as independently unseen evidence.",
        markdown_table(
            pd.DataFrame(folds)[
                [
                    "fold",
                    "phase",
                    "train_rows",
                    "test_rows",
                    "last_train_label",
                    "test_start",
                    "last_test_label",
                ]
            ]
        ),
        "## Incremental value of sentiment",
        summary + " " + verdict + " Positive gains favor adding sentiment. Direction gains "
        "are percentage points. Paired 95% percentile intervals use 500 circular moving-block resamples of the "
        "absolute loss difference. Block lengths exceed or equal the forecast horizon. Intervals are descriptive "
        "and unadjusted for testing multiple horizons. Fold wins count lower MAE.",
        markdown_table(gains),
        "## Out-of-sample forecast errors",
        "MAE, RMSE and mean signed error use price units. "
        "Both price-level and return R² are reported because a persistent price level can produce a high R² with "
        "little return predictability. Direction is exact sign agreement, including a zero-return forecast. "
        "Positive MAE skill indicates an improvement over random walk.",
        markdown_table(metrics),
        "![Model comparison and fold performance](figures/model_evaluation.png)",
        "![Actual versus predicted target closes](figures/predicted_actual.png)",
        "## Trading simulation",
        f"The one-bar forecast sets a long/cash position. Each training fold sets the "
        f"entry threshold to the larger of the median absolute one-bar training log return and two one-way costs. "
        f"Enter at the next bar open, liquidate at its close, with unit maximum exposure and no borrowing. "
        f"One-way cost is {cfg.cost_bps:g} bp including {cfg.slippage_bps:g} bp of slippage. "
        "Each active bar is one round trip, even for consecutive signals. The historical opening print is a fill "
        "assumption, especially for adjacent crypto bars. Buy-and-hold enters once and exits once per phase. "
        f"Sharpe uses sample standard deviation and {periods:g} periods per year. Sortino uses the root mean "
        "squared negative return across all bars. Both use a zero hurdle. Drawdown includes initial wealth. "
        "Gross and net are reported separately. Annualized growth rates extrapolate short samples.",
        "**Selected model, final holdout, net of costs**",
        markdown_table(chosen_risk),
        markdown_table(risk),
        "![Net equity and drawdown](figures/strategy_equity_drawdown.png)",
        "## Limitations and reproduction",
        "Capped sampling, sparse coverage, changing regimes and domain mismatch "
        "can swamp small sentiment effects. Repeated texts, multilingual posts, quoted prices and bots complicate "
        "interpretation. FinBERT scores in this retrospective reconstruction use a pinned model artifact distributed "
        "after the 2021 historical sample. Historical scores do not establish that this exact artifact was deployable then. "
        "Correlation does not identify information arrival or causation. A historical backtest "
        "cannot establish present market sentiment. News is retrieved separately for dated outlooks.",
        "The manifest records the exact feature order, model parameters, input and software hashes, provider runs, "
        "forecast cutoffs and engine version. The README gives the offline replay, public-source rebuild and "
        "credential-gated collection commands. See forecast_metrics.csv, fold_metrics.csv, incremental.csv, "
        "trades.csv and the forecast matrix for machine-readable results.",
    ]
    text = "\n\n".join(s for s in sections if s) + "\n"
    for file in ["evaluation.md", "evaluation_report.md"]:
        (out / file).write_text(text, encoding="utf-8")


def write_outlook(out, frame, news, forecasts, mode, audit, cfg, selected, errors):
    stamp = frame.index[-1]
    period = 24 if cfg.interval == "1h" else 7
    recent, previous = frame.tail(period), frame.iloc[-2 * period : -period]

    def average(g, column):
        return (g[column] * g.post_count).sum() / g.post_count.sum() if g.post_count.sum() else np.nan

    level, prev = average(recent, "sentiment"), average(previous, "sentiment")
    label = (
        "unavailable"
        if pd.isna(level)
        else "positive"
        if level > 0.05
        else "negative"
        if level < -0.05
        else "neutral"
    )
    momentum = (
        "unavailable"
        if pd.isna(level - prev)
        else "improving"
        if level - prev > 0.05
        else "deteriorating"
        if level - prev < -0.05
        else "stable"
    )
    fmt = lambda v: "unavailable" if pd.isna(v) else f"{v:.3f}"
    from .ingest import save_json
    from .news_context import select_news

    asof = pd.Timestamp.now(tz="UTC") if mode == "REAL_DATA" else stamp
    items, context = select_news(news, cfg, asof)
    save_json(out / "news_context.json", context)
    results = pd.DataFrame(forecasts)[["horizon", "kind", "price", "simple_return", "time"]]
    text = [
        f"# {cfg.asset} | {cfg.platform} sentiment outlook",
        f"**{mode.replace('_', ' ')}** · As of {stamp.isoformat()} "
        f"({stamp.tz_convert(cfg.timezone).isoformat()} exchange time).",
        f"**Computed observations.** Over the last {period} completed bars the sampled sentiment was **{label}** "
        f"({fmt(level)} versus {fmt(prev)} in the previous block). Momentum was **{momentum}**, using a ±0.05 band. "
        f"Sampled posts totalled {int(recent.post_count.sum())} versus {int(previous.post_count.sum())}. "
        f"Positive and negative shares were {fmt(average(recent, 'bull_fraction'))} and {fmt(average(recent, 'bear_fraction'))}. "
        f"The observed close was {frame.close.iloc[-1]:,.2f}, a {frame.close.iloc[-1] / frame.close.iloc[-period - 1] - 1:+.2%} "
        "change over the same price window.",
        f"**News context as of {asof.strftime('%Y-%m-%d %H:%M UTC')}**",
        "\n".join(items[-5:]) or "No dated, contemporaneously retrieved Tavily/Exa articles available.",
        f"**Predictive outlook.** Validation selected {selected}. Empirical holdout one-bar price MAE was "
        f"{errors:.2f}. These point estimates have no calibrated prediction interval.",
        markdown_table(results),
        "**Interpretation.** Coverage describes the collected sample. Empty intervals provide no evidence for a "
        "directional sentiment call. Historical projections apply only at their stated cutoff.",
    ]
    (out / "sentiment_outlook.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def write_dashboard(out, frame, predictions, curves, forecasts, mode, cfg):
    label = (
        "SYNTHETIC DEMO"
        if mode == "SYNTHETIC_DEMO"
        else "REAL HISTORICAL DATA"
        if mode == "REAL_HISTORICAL"
        else "Retrospective research"
    )
    price = make_subplots(specs=[[{"secondary_y": True}]])
    price.add_trace(
        go.Scatter(x=frame.index, y=frame.close, name=cfg.asset + " close", line={"color": "#51b9ed"})
    )
    price.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["sentiment_24h" if cfg.interval == "1h" else "sentiment_24d"],
            name="24-bar sentiment",
            line={"color": "#ffc168"},
        ),
        secondary_y=True,
    )
    price.update_yaxes(title_text="USD", secondary_y=False)
    price.update_yaxes(title_text="Sentiment", range=[-1, 1], secondary_y=True)
    last = frame.tail(28 * 24).copy()
    last["day"] = last.index.strftime("%Y-%m-%d")
    last["hour"] = last.index.hour
    last.loc[last.post_count == 0, "sentiment"] = np.nan
    matrix = last.pivot(index="day", columns="hour", values="sentiment").reindex(columns=range(24))
    heatmap = go.Figure(
        go.Heatmap(
            x=list(range(24)),
            y=matrix.index,
            z=matrix.to_numpy(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar={"thickness": 12, "len": 0.8},
            name="sentiment",
            hovertemplate="%{y} %{x}:00 UTC<br>sentiment %{z:.3f}<extra></extra>",
        )
    )
    heatmap.update_xaxes(title_text="Hour of day · UTC", tickmode="linear", dtick=4)
    projection = go.Figure(
        go.Scatter(
            x=[r["time"] for r in forecasts],
            y=[r["price"] for r in forecasts],
            mode="lines+markers+text",
            text=[f"n+{r['horizon']}" for r in forecasts],
            textposition="top center",
            name="Latest projection",
            line={"color": "#80dca0"},
        )
    )
    projection.update_yaxes(title_text="USD", tickformat=",.2f")
    mae_plot, equity = go.Figure(), go.Figure()
    colors = {
        "persistence": "#ffc168",
        "price_only": "#51b9ed",
        "sentiment_price": "#d99bea",
        "buy_hold": "#80dca0",
    }
    for name, g in predictions[predictions.phase == "holdout"].groupby("model"):
        mae = g.assign(error=(g.predicted_price - g.actual_price).abs()).groupby("horizon").error.mean()
        mae_plot.add_trace(
            go.Scatter(
                x=mae.index,
                y=mae.values,
                mode="lines+markers",
                name=name,
                line={"color": colors.get(name, "#ef9a76")},
            )
        )
    mae_plot.update_yaxes(title_text="MAE · USD")
    mae_plot.update_xaxes(title_text="Forecast horizon · bars", dtick=1)
    for name, g in curves[curves.phase == "holdout"].groupby("model"):
        equity.add_trace(
            go.Scatter(x=g.time, y=g.equity, name=name, line={"color": colors.get(name, "#ef9a76")})
        )
    equity.update_yaxes(title_text="Equity · initial 1")
    # Seven days keeps individual hourly bars legible; full history is in features.csv.
    counts = go.Figure(
        go.Bar(
            x=frame.tail(168).index,
            y=frame.tail(168).post_count,
            name="Unique posts",
            marker_color="#a79be7",
        )
    )
    charts = [
        ("Price and trailing 24-bar sentiment", price, False),
        ("Observed sentiment by UTC close", heatmap, False),
        ("Observed n+0 and forecasts n+1…n+5", projection, False),
        ("Forecast errors · final holdout", mae_plot, False),
        ("Out-of-sample equity · final holdout", equity, True),
        ("Sampled posts · last 168 bars", counts, True),
    ]
    sections = []
    for i, (title, figure, wide) in enumerate(charts):
        figure.update_layout(
            template="plotly_dark",
            height=370,
            paper_bgcolor="#151c28",
            plot_bgcolor="#151c28",
            font={"family": "Arial", "size": 11},
            margin={"t": 20, "b": 85, "l": 60, "r": 50},
            legend={"orientation": "h", "y": -0.24, "font": {"size": 10}},
            autosize=True,
        )
        markup = figure.to_html(
            full_html=False,
            include_plotlyjs=True if i == 0 else False,
            div_id=f"chart-{i}",
            config={"responsive": True, "displaylogo": False},
        )
        sections.append(
            f'<section class="card {"wide" if wide else ""}"><h2>{html.escape(title)}</h2>{markup}</section>'
        )
    banner = (
        "Generated fixture data. These charts do not describe current BTC sentiment or real market performance."
        if mode == "SYNTHETIC_DEMO"
        else "Retrospective observational research. Review source coverage and evaluation assumptions before use."
    )
    if mode == "REAL_HISTORICAL":
        banner = "Historical social data aligned with observed prices. This dated backtest does not describe current sentiment or satisfy recent Apify/Tavily collection."
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment and price research</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1019;color:#e3eaf6;font-family:Arial,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}.badge{color:#ffc168;font-size:12px;font-weight:bold;letter-spacing:1px}
h1{font-size:28px;margin:12px 0}p{color:#b2bfd2;line-height:1.5;max-width:1000px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:26px}
.card{min-width:0;background:#151c28;border:1px solid #273349;border-radius:12px;overflow:hidden;padding:16px 4px 0}
h2{font-size:16px;margin:0 16px 10px;font-weight:500}.wide{grid-column:1/-1}
@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr;gap:16px}h1{font-size:23px}.card{padding-top:16px}}
</style></head><body><main>"""
    page += f'<div class="badge">{html.escape(label)}</div><h1>{cfg.asset} · {cfg.platform} sentiment</h1><p>{html.escape(banner)}</p>'
    page += f"<p>{cfg.interval} bars · n+0 is observed · n+1…n+5 are forecasts. UTC throughout. Matrix: red is negative, blue is positive; blank cells have no sampled posts. Hover to inspect, drag to zoom, and click legends to compare.</p>"
    page += '<div class="grid">' + "\n".join(sections) + "</div></main></body></html>"
    (out / "dashboard.html").write_text(page, encoding="utf-8")
