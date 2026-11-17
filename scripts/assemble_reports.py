"""Publish the review copies and a checksum manifest without copying raw posts or credentials."""

import json
import shutil
from pathlib import Path

import pandas as pd

from xsentiment.pipeline import digest


def assemble(base=Path("reports")):
    historical, supplementary, current = [base / d for d in ("historical", "stocktwits", "current")]
    hist = json.loads((historical / "manifest.json").read_text())
    stock = json.loads((supplementary / "manifest.json").read_text())
    recent = json.loads((current / "forecast.json").read_text())
    news = json.loads((current / "news_context.json").read_text())
    text = (historical / "evaluation_report.md").read_text(encoding="utf-8")
    gains = pd.read_csv(historical / "incremental.csv")
    d_gains = pd.read_csv(supplementary / "incremental.csv")
    d_data = json.loads((supplementary / "data_quality.json").read_text())
    d_folds = json.loads((supplementary / "folds.json").read_text())
    d_hold = d_folds[-1]["test_rows"]
    wf_wins = (gains.loc[gains.phase == "walk_forward", "mae_gain_usd"] > 0).sum()
    final_wins = (gains.loc[gains.phase == "holdout", "mae_gain_usd"] > 0).sum()
    d_wins = (d_gains.loc[d_gains.phase == "holdout", "mae_gain_usd"] > 0).sum()
    text += f"""
## Other experiments in this submission

| Experiment | Evidence available | Finding |
|---|---|---|
| A · Recent X and price | 180 days of recent Yahoo prices. Current news providers: {", ".join(news["providers"]) or "none"}. Authenticated X collection pending. | Current X sentiment and its incremental value are unmeasured. |
| B · Historical X and price | FinBERT-scored Twitter sample and Coinbase hourly bars | Adding sentiment improved Ridge MAE at {wf_wins}/5 development horizons and {final_wins}/5 holdout horizons. |
| C · Historical and recent X | No validated recent X corpus overlapping the historical source transition | No pooled result or evidence of cross-period generalization. |
| D · Supplementary StockTwits | {d_data["price_rows"]} daily Coinbase bars and {d_data["unique_text_posts_in_window"]:,} aligned self-labelled posts | {d_data["labeled_observations"]} labelled origins, {len(d_folds) - 1} development folds, {d_hold} holdout origins. Validation selected {stock["selected_model"]}. Adding StockTwits labels reduced holdout Ridge MAE at {d_wins}/5 horizons. |

The independent recent price-only control selected {recent["selected_model"]} and supplies the latest available projections. See the [current outlook](sentiment_outlook.md), [StockTwits evaluation](stocktwits/evaluation_report.md), [catalogue review](dataset_discovery_report.md) and [historical run tables](historical/forecast_metrics.csv).
"""
    (base / "evaluation_report.md").write_text(text, encoding="utf-8")
    shutil.copyfile(current / "sentiment_outlook.md", base / "sentiment_outlook.md")
    shutil.copytree(historical / "figures", base / "figures", dirs_exist_ok=True)
    blocked_names = {"features.csv", "dashboard.html", "model.joblib"}
    outputs = {}
    for path in base.rglob("*"):
        if (
            path.is_file()
            and path.name not in blocked_names
            and not path.name.startswith("input_")
            and path.name != "reproducibility_manifest.json"
        ):
            outputs[path.relative_to(base).as_posix()] = digest(path)
    result = dict(
        asset=hist["config"]["asset"],
        experiment_b=hist["id"],
        experiment_d=stock["id"],
        recent_price_control=recent["model_version"],
        recent_cutoff=recent["data_cutoff"],
        historical_cutoff=hist["feature_cutoff"],
        sentiment_engine=hist["engine"],
        finbert_revision="0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
        feature_order=hist["features"],
        models=hist["models"],
        libraries=hist["libraries"],
        selection_scores=hist["selection_log_rmse"],
        apify_actor_default="apidojo~tweet-scraper",
        apify_template="config/apify_actor_input.example.json",
        apify_run_ids=[],
        tavily_live="tavily" in news["providers"],
        exa_live="exa" in news["providers"],
        news_articles_stored=news["articles_stored"],
        providers=["Kaggle", "Coinbase", "Yahoo Finance", "NYU public S3", *news["providers"]],
        experiments={
            "A": "recent X unavailable",
            "B": "executed",
            "C": "recent X unavailable",
            "D": "executed separately",
        },
        local_run_files_excluded_from_git=["input_*.csv", "features.csv", "dashboard.html", "model.joblib"],
        published_sha256=outputs,
    )
    (base / "reproducibility_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Indexed {len(outputs)} published report artifacts")


if __name__ == "__main__":
    assemble()
