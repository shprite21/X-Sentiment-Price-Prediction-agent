"""Experiment C. Build all labels inside each era, then augment recent training with the older segment."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from xsentiment import storage as db
from xsentiment.config import Config
from xsentiment.features import make_targets
from xsentiment.ingest import save_json
from xsentiment.model import backtest, evaluate, fit_model, forecast_metrics, incremental, select_model
from xsentiment.pipeline import digest, records
from xsentiment.report import markdown_table


def inputs(path):
    manifest = json.loads((path / "manifest.json").read_text())
    plain = path / "features.csv"
    content = (
        plain.read_bytes() if plain.exists() else gzip.decompress((path / "features.csv.gz").read_bytes())
    )
    if hashlib.sha256(content).hexdigest() != manifest["artifact_sha256"]["features.csv"]:
        raise ValueError("Feature checksum mismatch")
    if digest(path / "forecast_metrics.csv") != manifest["artifact_sha256"]["forecast_metrics.csv"]:
        raise ValueError("Forecast metric checksum mismatch")
    frame = pd.read_csv(io.BytesIO(content), index_col="time", float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return manifest, frame


def compare(history, recent, store, out):
    old, b = inputs(history)
    new, a = inputs(recent)
    if old["experiment"] != "B" or new["experiment"] != "A":
        raise ValueError("Comparison requires independently validated X runs B and A")
    if any(
        old["config"][k] != new["config"][k]
        for k in ("asset", "interval", "exchange", "timezone", "availability")
    ):
        raise ValueError("Asset, bar convention or availability policy changed")
    if old["features"] != new["features"] or old["engine"] != new["engine"]:
        raise ValueError("Both eras require the same feature order and sentiment engine")
    cfg = Config(**new["config"])
    pc, ac = new["price_only_features"], new["features"]
    pred, folds = evaluate(a, pc, ac, cfg, earlier_frame=b)
    selected, selection = select_model(pred)
    metric, fold_metric = forecast_metrics(pred), forecast_metrics(pred, True)
    gain = incremental(pred, fold_metric, cfg.seed)
    risk, curves = backtest(pred, cfg, new["periods_per_year"])
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("predictions", pred),
        ("forecast_metrics", metric),
        ("fold_metrics", fold_metric),
        ("incremental", gain),
        ("strategy_metrics", risk),
        ("strategy_returns", curves),
        ("trades", curves[(curves.position > 0) & (curves.model != "buy_hold")]),
    ]:
        data.to_csv(out / (name + ".csv"), index=False)
    save_json(out / "folds.json", folds)
    # Direct transport: fit only to 2021. Reuse the actual recent origins and target prices.
    y_old = make_targets(b).dropna()
    stale = []
    for name, cols in [("price_only", pc), ("sentiment_price", ac)]:
        model = fit_model(b.loc[y_old.index, cols], y_old, cfg.ridge_alpha, name, cfg.seed)
        sample = pred[pred.model == name].copy()
        origins = pd.Index(sample.time.unique()).sort_values()
        with threadpool_limits(limits=2):
            forecasts = pd.DataFrame(
                model.predict(a.loc[origins, cols]), index=origins, columns=range(1, 6)
            )
        for h in range(1, 6):
            mask = sample.horizon == h
            sample.loc[mask, "predicted_return"] = sample.loc[mask, "time"].map(forecasts[h])
        sample["predicted_price"] = sample.origin_price * np.exp(sample.predicted_return)
        stale.append(sample)
    transported = forecast_metrics(pd.concat(stale, ignore_index=True))
    transported.to_csv(out / "historical_only_on_recent.csv", index=False)
    original = pd.read_csv(recent / "forecast_metrics.csv")
    merged = metric.merge(
        original, on=["phase", "model", "horizon"], suffixes=("_pooled", "_recent"), validate="one_to_one"
    )
    merged["mae_gain_from_archive"] = merged.mae_usd_recent - merged.mae_usd_pooled
    paired = merged[
        ["phase", "model", "horizon", "mae_usd_recent", "mae_usd_pooled", "mae_gain_from_archive"]
    ]
    paired.to_csv(out / "archive_increment.csv", index=False)
    distribution = []
    for era, root, x in [("Historical", history, b), ("Recent", recent, a)]:
        source = pd.read_csv(root / "source_distributions.csv")
        distribution.append(
            {
                "era": era,
                "scored_posts": int(source.posts.sum()),
                "coverage": x.has_posts.mean(),
                "post_mean": float(np.average(source["mean"], weights=source.posts)),
                "positive_share": (x.bull_fraction * x.post_count).sum() / x.post_count.sum(),
                "negative_share": (x.bear_fraction * x.post_count).sum() / x.post_count.sum(),
                "neutral_share": (x.neutral_fraction * x.post_count).sum() / x.post_count.sum(),
            }
        )
    distributions = pd.DataFrame(distribution)
    distributions.to_csv(out / "period_distributions.csv", index=False)
    hold = paired.query("phase=='holdout' and model=='sentiment_price'")
    transported_h = transported.query("phase=='holdout' and model=='sentiment_price'")[
        ["horizon", "mae_usd", "r2_return"]
    ]
    pooled_wins = int((hold.mae_gain_from_archive > 0).sum())
    report = [
        "# BTC-USD | Experiment C across two market periods",
        (
            f"Adding the archived training segment reduced the sentiment Ridge's holdout MAE at **{pooled_wins}/5** "
            f"horizons relative to training only on the recent sample. Validation selected **{selected}** among the four "
            "pooled candidates. The recent X run remains the primary assessment."
        ),
        "## Cohorts and target construction",
        (
            "Each period was independently aligned, scored and warmed up on its own complete price grid. "
            "The last five origins from each training segment require matured within-segment labels. No feature, "
            "target or synthetic holding return crosses the gap between 2021 and 2026. Each recent fold receives "
            "the same historical observations and only its purged expanding block of recent training origins. "
            "Imputation and scaling are fitted again on this union inside every fold. Recent test origins, costs, "
            "models and baseline match Experiment A. The historical Coinbase series and recent Yahoo series use "
            "different price providers, and the two X sampling mechanisms differ."
        ),
        "## Sentiment distributions",
        markdown_table(distributions),
        (
            "Shares and coverage refer to scored samples. Topic mix, ranking, post caps and language selection "
            "can change them independently of market sentiment."
        ),
        "## Does historical training help on recent dates?",
        "Positive dollar gains indicate lower absolute error from adding the historical segment.",
        markdown_table(hold),
        (
            "As a second test, both Ridge models were frozen after fitting only historical labels and applied to "
            "the same recent origins. The sentiment Ridge holdout results are below. All five horizons and both "
            "Ridges are available in historical_only_on_recent.csv."
        ),
        markdown_table(transported_h),
        "## Incremental sentiment effect within pooled training",
        markdown_table(gain),
        "## All horizons and trading results",
        markdown_table(metric),
        markdown_table(risk),
        (
            "Five-bar purges isolate validation and final holdout labels. The selected specification was determined "
            "on recent validation returns. There are only two sampled eras, so this check cannot isolate platform "
            "drift from price-provider, coverage, regime and sampling changes."
        ),
    ]
    (out / "evaluation_report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    run_id = hashlib.sha256(
        (
            old["id"]
            + new["id"]
            + digest(Path(__file__))
            + digest(Path(__file__).resolve().parents[1] / "src/xsentiment/model.py")
        ).encode()
    ).hexdigest()
    manifest = {
        "experiment": "C",
        "mode": "REAL_SEGMENTED_X",
        "id": run_id,
        "historical_run": old["id"],
        "recent_run": new["id"],
        "selected_model": selected,
        "selection_log_rmse": selection,
        "archive_rows": len(y_old),
        "recent_initial_train_rows": folds[0]["recent_train_rows"],
        "recent_test_range": [folds[0]["test_start"], folds[-1]["test_end"]],
        "evaluation_basis": "recent origins only",
        "holdout_pooled_ridge_mae_wins": pooled_wins,
        "artifact_sha256": {p.name: digest(p) for p in out.glob("*") if p.name != "manifest.json"},
    }
    save_json(out / "manifest.json", manifest)
    with store.engine.begin() as con:
        store.upsert(db.experiments, [{"id": run_id, "manifest": manifest}], con)
        store.upsert(db.forecasts, records(pred.assign(experiment=run_id)), con)
        store.upsert(
            db.metrics,
            [
                {
                    "experiment": run_id,
                    "phase": r["phase"],
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "values": r,
                }
                for r in records(metric)
            ],
            con,
        )
    print("Combined experiment", selected, "archive augmentation holdout MAE wins", pooled_wins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=Path("reports/historical"))
    parser.add_argument("--recent", type=Path, default=Path("reports/recent_x"))
    parser.add_argument("--out", type=Path, default=Path("reports/combined"))
    parser.add_argument("--db", default="data/database/recent-yahoo.sqlite")
    args = parser.parse_args()
    compare(args.historical, args.recent, db.Store(args.db), args.out)
