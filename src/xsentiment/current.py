"""Recent price-only benchmark when the recent X experiment cannot be estimated."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from . import storage as db
from .config import Config
from .features import build_features
from .ingest import save_json
from .model import evaluate, forecast, forecast_metrics, select_model
from .pipeline import digest, records, run
from .report import markdown_table


def write_current(research: Path, store, out: Path):
    source = json.loads((research / "manifest.json").read_text(encoding="utf-8"))
    cfg = Config(**source["config"])
    store.bind(cfg)
    prices = store.frame("prices").sort_values("time")
    posts = store.frame("posts")
    if not posts.empty and (posts.source == "apify").any():
        return run(store, out, cfg, experiment="A")
    out.mkdir(parents=True, exist_ok=True)
    if prices.empty:
        raise ValueError("Current outlook requires completed price bars")
    stamps = pd.to_datetime(prices.time, utc=True, format="mixed")
    prices = prices.loc[stamps > stamps.max() - pd.Timedelta(days=cfg.days)]
    f, pc, _, audit = build_features(prices, posts, cfg, price_only=True)
    predictions, folds = evaluate(f, pc, pc, cfg, names=("persistence", "price_only"))
    selected, scores = select_model(predictions)
    metrics = forecast_metrics(predictions)
    model, projections = forecast(f, pc, cfg, selected)
    prices.to_csv(out / "input_prices.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    save_json(out / "folds.json", folds)
    version = hashlib.sha256(
        (
            digest(out / "input_prices.csv")
            + selected
            + json.dumps(cfg.to_dict(), sort_keys=True)
            + digest(Path(__file__).with_name("model.py"))
            + digest(Path(__file__).with_name("features.py"))
        ).encode()
    ).hexdigest()
    time = db.timestamp()
    origin = f.index[-1].isoformat()
    info = dict(
        experiment="recent_price_only_control",
        asset=cfg.asset,
        selected_model=selected,
        model_version=version,
        training_timestamp=time,
        data_cutoff=origin,
        feature_cutoff=origin,
        prediction_timestamp=time,
        training_last_origin=f.index[-6].isoformat(),
        selection_log_rmse=scores,
        forecasts=projections,
        price_source=prices.source.unique().tolist(),
        input_sha256=digest(out / "input_prices.csv"),
        model_config=cfg.to_dict(),
        features=pc,
        x_coverage="unavailable",
    )
    save_json(out / "forecast.json", info)
    joblib.dump({"model": model, "features": pc, "version": version}, out / "model.joblib")
    with store.engine.begin() as con:
        store.upsert(db.forecasts, records(predictions.assign(experiment=version)), con)
        store.upsert(
            db.forecasts,
            [
                dict(
                    experiment=version,
                    model=selected,
                    time=origin,
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
                dict(experiment=version, phase=r["phase"], model=r["model"], horizon=r["horizon"], values=r)
                for r in records(metrics)
            ],
            con,
        )
        store.upsert(db.experiments, [dict(id=version, manifest=info)], con)
    news = store.frame("news")
    contexts = []
    cited = []
    publishers = set()
    terms = {term.casefold() for term in (cfg.company, cfg.ticker, *cfg.keywords) if term}
    for row in news.sort_values("published_at", ascending=False).itertuples():
        published = pd.to_datetime(row.published_at, utc=True, errors="coerce")
        title = " ".join(str(row.title).split()).replace("[", "(").replace("]", ")")
        if not any(term in title.casefold() for term in terms):
            continue
        if (
            pd.Timestamp(time) - pd.Timedelta(days=7) <= published <= pd.Timestamp(time)
            and pd.Timestamp(row.retrieved_at) <= pd.Timestamp(time)
            and row.publisher not in publishers
            and str(row.url).startswith(("https://", "http://"))
        ):
            publishers.add(row.publisher)
            contexts.append(f"- {published.date()} · {row.publisher}. [{title}]({row.url})")
            cited.append(
                dict(
                    provider=row.source,
                    publisher=row.publisher,
                    headline=title,
                    url=row.url,
                    published_at=row.published_at,
                    retrieved_at=row.retrieved_at,
                )
            )
            if len(cited) == 3:
                break
    news_audit = dict(
        report_time=time,
        providers=sorted(news.source.unique().tolist()),
        articles_stored=len(news),
        articles_cited=cited,
        collection=store.metadata().get("news_collection"),
        predictive_news_columns=[],
    )
    save_json(out / "news_context.json", news_audit)
    table = pd.DataFrame(projections)[["horizon", "kind", "price", "simple_return", "time"]]
    table.to_csv(out / "forecast.csv", index=False)
    last = metrics.query("phase=='holdout' and model==@selected and horizon==1").iloc[0]
    text = [
        f"# {cfg.asset} | Latest available market outlook",
        f"**Price cutoff {origin}.**",
        "**X sentiment and social activity.** Recent X collection is unavailable. Sentiment, momentum, "
        "sampled post count and positive/negative balance are unknown. The historical X model cannot "
        "support a current sentiment call.",
        f"**Price context.** The latest completed close was {f.close.iloc[-1]:,.2f}, "
        f"{f.close.iloc[-1] / f.close.iloc[-25] - 1:+.2%} over 24 bars. Source: {', '.join(info['price_source'])}.",
        f"**News context at {time[:16].replace('T', ' ')} UTC.** These publisher headlines were retrieved separately from the forecast.",
        "\n".join(contexts[:5]) or "No contemporaneous Tavily/Exa articles were collected.",
        f"**Price-only forecast.** On {audit['price_rows']:,} recent bars, {selected} had the lower pooled "
        f"validation log-return RMSE among random walk and price-only Ridge. Five-bar purges separated "
        f"training, validation labels and holdout. Final one-bar MAE was {last.mae_usd:.2f} and directional "
        f"accuracy was {last.direction_accuracy:.1%}. This is a separate control experiment. The five "
        "projections are point estimates and recent X contributions remain unmeasured.",
        markdown_table(table),
    ]
    (out / "sentiment_outlook.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "price_cutoff": origin, "experiment": info["experiment"]}))
