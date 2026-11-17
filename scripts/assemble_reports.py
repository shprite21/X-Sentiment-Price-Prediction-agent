"""Publish the primary recent experiment alongside the historical and transfer studies."""

import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

from xsentiment.pipeline import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive_features(directory):
    """Share bar-level aggregates so a reviewer can refit without redistributing individual posts."""
    plain = directory / "features.csv"
    if plain.is_file():
        with (directory / "features.csv.gz").open("wb") as out:
            with (
                gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as gz,
                plain.open("rb") as source,
            ):
                shutil.copyfileobj(source, gz)


def assemble(base=Path("reports")):
    dirs = {
        "A": base / "recent_x",
        "B": base / "historical",
        "C": base / "combined",
        "D": base / "stocktwits",
    }
    runs = {label: read(folder / "manifest.json") for label, folder in dirs.items()}
    a, b, c, d = (runs[key] for key in "ABCD")
    recent = dirs["A"]
    news = read(recent / "news_context.json")
    billing = read(Path("reference/audit/apify_costs.json"))
    charged = sum(row["charged_events_usd"] for row in billing["runs"])
    b_gain = pd.read_csv(dirs["B"] / "incremental.csv")
    b_dev = int((b_gain.loc[b_gain.phase == "walk_forward", "mae_gain_usd"] > 0).sum())
    b_hold = int((b_gain.loc[b_gain.phase == "holdout", "mae_gain_usd"] > 0).sum())
    for key in "AB":
        archive_features(dirs[key])
    text = (recent / "evaluation_report.md").read_text(encoding="utf-8")
    text = text.replace(
        "# BTC-USD | X research", "# BTC-USD | Does X sentiment improve hourly forecasts?", 1
    )
    text += f"""
## Evidence across periods and platforms

| Experiment | Independent comparison | Outcome |
|---|---|---|
| A · Recent X | 401 FinBERT-scored posts and 4,320 hourly Yahoo closes, May–November 2026 | Adding sentiment increased Ridge MAE and RMSE at all five horizons in development and holdout. Validation selected {a["selected_model"]}. |
| B · Historical Twitter | 20,243 scored posts and 4,320 hourly Coinbase closes, February–August 2021 | Adding sentiment reduced Ridge MAE at {b_dev}/5 development horizons and {b_hold}/5 final holdout horizons. |
| C · Historical and recent X | Add the older matured labels to each recent training fold, with features and targets constructed within each era | Archive augmentation improved the sentiment Ridge's recent holdout MAE at {c["holdout_pooled_ridge_mae_wins"]}/5 horizons. Pooled validation selected {c["selected_model"]}. |
| D · StockTwits | Separate daily BTC.X self-labelled sample and 180 Coinbase daily closes | Validation selected {d["selected_model"]}. The final holdout contained 17 origins. |

Two Apify runs reported USD {charged:.2f} in charges. Tavily returned {news["articles_stored"]} articles for the outlook. Exa was exercised with a test double. The cap and ranking limit the interpretation of recent post counts and their changes. The study measures the incremental value of this sampled corpus.

[Latest outlook](sentiment_outlook.md) · [Recent five-horizon forecast matrix](recent_x/forecast_horizon_matrix.csv) · [Full recent run](recent_x/evaluation_report.md) · [Historical X](historical/evaluation_report.md) · [Cross-period distributions and forecasts](combined/evaluation_report.md) · [StockTwits](stocktwits/evaluation_report.md) · [Price-only control](current/sentiment_outlook.md) · [Data discovery](dataset_discovery_report.md) · [Reproduction commands](../README.md).
"""
    (base / "evaluation_report.md").write_text(text, encoding="utf-8")
    shutil.copyfile(recent / "sentiment_outlook.md", base / "sentiment_outlook.md")
    shutil.copytree(recent / "figures", base / "figures", dirs_exist_ok=True)
    excluded = {"features.csv", "dashboard.html", "model.joblib", "reproducibility_manifest.json"}
    outputs = {
        p.relative_to(base).as_posix(): digest(p)
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.name not in excluded and not p.name.startswith("input_")
    }
    result = {
        "asset": a["config"]["asset"],
        "experiments": {
            k: {
                "run_id": m["id"],
                "selected_model": m["selected_model"],
                "manifest": (dirs[k] / "manifest.json").relative_to(base).as_posix(),
            }
            for k, m in runs.items()
        },
        "recent_cutoff": a["feature_cutoff"],
        "historical_cutoff": b["feature_cutoff"],
        "recent_price_control": read(base / "current/forecast.json")["model_version"],
        "data_ranges": {
            "A": a["date_range"],
            "B": b["date_range"],
            "C": c["recent_test_range"],
            "D": d["date_range"],
        },
        "sentiment_engine": a["engine"],
        "finbert_revision": "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
        "feature_order": a["features"],
        "models": a["models"],
        "libraries": a["libraries"],
        "feature_config": a["config"],
        "selection_scores": a["selection_log_rmse"],
        "apify_actor_id": a["apify"]["actor_id"],
        "apify_template": "config/apify_coregent.json",
        "apify_run_ids": [r["id"] for r in a["apify"]["runs"]],
        "actor_inputs": [r["input"] for r in a["apify"]["runs"]],
        "apify_total_configured_cap_usd": a["apify"]["budget_cap_usd"],
        "apify_reported_charges_usd": round(charged, 2),
        "apify_cost_audit": "../reference/audit/apify_costs.json",
        "tavily_live": "tavily" in news["providers"],
        "exa_live": "exa" in news["providers"],
        "news_articles_stored": news["articles_stored"],
        "providers": ["Apify", "Kaggle", "Coinbase", "Yahoo Finance", "NYU public S3", *news["providers"]],
        "local_run_files_excluded_from_git": [
            "input_*.csv",
            "features.csv",
            "dashboard.html",
            "model.joblib",
        ],
        "published_sha256": outputs,
    }
    (base / "reproducibility_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Indexed {len(outputs)} published report artifacts")


if __name__ == "__main__":
    assemble()
