# CrowdWisdomTrading | X sentiment and price forecasting

**Did X sentiment help?** Adding FinBERT-scored X posts to the price-only Ridge increased BTC-USD price MAE and RMSE at every one- to five-hour horizon, both across the five recent walk-forward folds and on the final holdout. One-hour holdout MAE rose from **$214.50 to $218.15** and directional accuracy fell from **52.98% to 49.51%**. Validation selected the random walk. Only 8.6% of usable hours contained a sampled post.

| Research output | Where to look |
|---|---|
| Methods, all four models, five horizons, trading and limitations | [Evaluation report](reports/evaluation_report.md) |
| Latest available close, current sentiment coverage and five validated projections | [Current outlook](reports/sentiment_outlook.md) |
| All 12 catalogue entries, sizes, access verification, date integrity and selection | [Dataset discovery](reports/dataset_discovery_report.md) |
| Recent forecasts and ablation results | [Forecast matrix](reports/recent_x/forecast_horizon_matrix.csv) · [Paired comparisons](reports/recent_x/incremental.csv) |
| Static research figures | [Figures](reports/figures/) |
| Separate daily StockTwits experiment | [Supplementary report](reports/stocktwits/evaluation_report.md) |
| Provenance, environment and checksums | [Submission manifest](reports/reproducibility_manifest.json) · [Verification](docs/verification.md) · [Private credential checklist](docs/submission.md) |

![Recent model comparison](reports/figures/model_evaluation.png)

## Evidence

Experiment A paired **401 scored X posts** with **4,320 hourly Yahoo BTC-USD bars** from 21 May through 17 November 2026. Two Apify runs returned 420 rows. Of the 414 records inside the query windows, 11 repeated cleaned text and two failed relevance or text checks. Five expanding validation blocks evaluated 1,402 origins. The final holdout evaluated 721. Five-bar purges protected the training and model-selection boundaries. The selected random walk generated zero qualifying trades, a zero return and zero drawdown. Both risk ratios are undefined. Tavily supplied 10 articles for the outlook.

Experiment B used **20,243 scored posts**, 4,320 actual hourly candles and 4,243 labelled origins between February and August 2021. Of the 4,248 feature rows after warmup, 3,103 lacked sampled posts. Five expanding validation blocks produced 1,402 OOS origins, followed by a separately purged 721-origin holdout. FinBERT probabilities, native StockTwits labels and VADER lexical weights are identified separately.

The selected gradient-boosting long/cash rule made 27 development trades. After 10 bp per side, it returned **3.00%**, with annualized Sharpe **0.72**, Sortino **1.03** and maximum drawdown **−7.45%**. It made zero final-holdout trades. Holdout return and drawdown were zero, and both ratios were undefined.

Experiment C adds 4,243 matured historical training origins to every recent fold. The pooled sentiment Ridge worsened recent holdout MAE at all five horizons. Pooling reduced boosting MAE at all five horizons relative to recent-only boosting, and pooled validation selected boosting. Historical-only Ridge transfer, sentiment distributions and all paired comparisons are in the [cross-period study](reports/combined/evaluation_report.md). Experiment D uses a separate daily StockTwits sample.

## Install and verify offline

Python 3.12 was used. On Windows, replace `python` below with `.\.venv\Scripts\python.exe` after creating the environment. On POSIX, use `.venv/bin/python` or activate the environment.

```sh
python -m venv .venv
# Activate the environment or use its interpreter for the remaining commands.
python -m pip install -e .
python -m xsentiment doctor
python -m unittest discover -s tests -v
python -m xsentiment demo
python scripts/verify_run.py runs/demo
```

The seeded demo generates synthetic fixtures, uses VADER, and performs no network requests. It runs the same database, feature, model, trading and reporting path. `run` and `run-all` both execute feature building, training, walk-forward evaluation, forecasting and reporting as one consistent experiment. Use a fresh demo database and output path for a second demo. The demo's metrics have no empirical market interpretation.

## Verify the actual models without API keys

The published gzip archives contain price and sentiment aggregates for each bar. The replay verifies the feature checksum, fits all four models and compares every out-of-sample predicted return and label with the recorded result.

```sh
python scripts/replay_models.py reports/recent_x
python scripts/replay_models.py reports/historical
python scripts/compare_periods.py --out runs/reproduced-c --db data/database/compare.sqlite
python scripts/verify_run.py runs/reproduced-c
```

## Reproduce historical X

The source is the publisher's original timestamped Bitcoin main CSV in its v54 ZIP. The Google Doc is a catalogue. The helper streams the ZIP to disk, verifies its SHA-256 and retrieves the pinned local financial model. Model weights and individual tweet text are excluded from Git. The checked-in 15 public Coinbase response pages reproduce the historical price feed offline.

```sh
python -m pip install -e ".[finbert]"
python scripts/fetch_research_inputs.py --live
python -m xsentiment prepare-history data/downloads/bitcoin-tweets-v54.zip --db data/database/experiment-b.sqlite
python -m xsentiment ingest --only prices --provider coinbase --cached --raw reference/coinbase --start 2021-02-06 --end 2021-08-05 --db data/database/experiment-b.sqlite
python -m xsentiment --config config/historical.toml run-all --experiment B --db data/database/experiment-b.sqlite --out runs/reproduced-b
python scripts/verify_run.py runs/reproduced-b
```

FinBERT uses the ProsusAI revision recorded in the manifest. CPU inference batches up to 32 texts at a time and caches probabilities by text and model/tokenizer hashes. The pinned local artifact was distributed after the evaluated trading period. This is a retrospective text-measurement experiment. A lightweight run omits `--config config/historical.toml` to select VADER. Use a distinct output path when changing engines.

For any completed local run, the following verifies the canonical input file hashes and rebuilds everything without provider requests. A FinBERT replay needs the already-downloaded local weights. Published subsets contain derived outputs; use the reconstruction above to restore individual input snapshots. Collection and fit timestamps and their provenance hashes can differ across fresh acquisitions.

```sh
python -m xsentiment replay runs/reproduced-b --db data/database/replay-b.sqlite --out runs/replay-b
```

## Collect recent X, prices and news

Copy `.env.example` to a local `.env`. Set APIFY_TOKEN and TAVILY_API_KEY, and optionally EXA_API_KEY. For the published experiment set `APIFY_ACTOR_ID=coregent~x-tweet-scraper` and `APIFY_INPUT_FILE=config/apify_coregent.json`. This template uses `{dated_queries}` for two-day search strata. The original `apidojo~tweet-scraper` template is available for accounts with access. [Coregent Actor and input schema](https://apify.com/coregent/x-tweet-scraper/input-schema) · [SDK references](docs/data_sources.md).

To restore the original X sample from the two recorded datasets, use an Apify credential authorized to read those datasets. The restore command uses read-only SDK methods. The price command below requests the frozen price window. The news command retrieves context as of its execution date. The original cited headlines and timestamps are in the published outlook.

```sh
python scripts/restore_apify_runs.py --live --db data/database/restored-a.sqlite
python -m xsentiment ingest --only prices --provider yahoo --days 180 --end 2026-11-17T12:00Z --live --db data/database/restored-a.sqlite
python -m xsentiment ingest --only news --live --db data/database/restored-a.sqlite
python -m xsentiment --config config/historical.toml run --experiment A --db data/database/restored-a.sqlite --out runs/restored-a
```

To commission a **new paid scrape**, set the Actor variables above and explicitly run the following with your intended cutoff and total budget. The published run divided USD 0.90 between two server-capped requests. Apify subsequently reported USD 0.84 for 420 results. Rank, deletions and provider updates can change a fresh sample. Raw pages and run IDs are saved for resumption. An ambiguous start must be reconciled against the provider run ledger before retrying.

```sh
python -m xsentiment ingest --only x --days 180 --end 2026-11-17T12:00Z --window-days 92 --query-days 2 --items-per-run 225 --budget-usd 0.90 --live --db data/database/new-collection.sqlite
```

With prices alone, a separate recent control evaluates random walk and price-only Ridge on recent expanding folds and fits the selected model on matured recent labels. The command writes the observed reference, five projected closes, UTC target times, empirical holdout error and the X/news coverage status.

```sh
python -m xsentiment current --research reports/historical --db data/database/recent.sqlite --out runs/current
```

## Import another asset or dataset

Edit a TOML `[research]` section using the fields in [Config](src/xsentiment/config.py). Set asset, ticker, company, keywords, exchange, timezone, interval, optional start/end, availability, min_train, test_size, holdout and embargo. Use a distinct database per asset and bar definition. The XNYS calendar with America/New_York supports session opens, holidays, DST and partial closes. Use `24/7` for crypto. Hourly (`1h`) and daily (`1d`) Yahoo bars are supported. Missing scheduled bars stop the analysis. Download a suitable frequency if a provider cannot serve older intraday history.

```sh
python -m xsentiment --config config/historical.toml import-posts data/raw/export.csv --dataset my-archive-v1 --license CC0-1.0 --db data/database/import.sqlite
```

Supply a genuine, asset-specific export at the path shown. CSV and JSONL are streamed, and JSON arrays are intended for small exports. Use the source's actual license. The importer reads text/full_text/content, createdAt/created_at/date/timestamp and id/id_str. UTC Unix seconds and milliseconds are supported. Naive dates mean UTC; convert a local-time source with its documented timezone before import. Nested Apify author metadata and flat author fields are supported. For unsourced post IDs, preserve the null original ID and create a stable content key. Optional `available_at` delays feature use. Publication-time reconstruction is explicit; `availability="first_seen"` additionally requires historical collector observations. Repeated imports retain the first acquisition. A historical price request without an explicit window extends its start to stored post history. Mixed assets must be partitioned by their actual source tags before import.

Select repeatable source subsets with repeated `--dataset` arguments to `run`. A uses Apify X, B uses historical X, single-database C requires contiguous source coverage (use the segmented comparison above for non-adjacent eras), and D requires a separate StockTwits configuration. The model only sees the explicit price and sentiment column lists. Target columns, news, exchange volume and retrospective engagement have no route into the predictors.

## Reproduce StockTwits separately

```sh
python scripts/sample_stocktwits.py
python scripts/prepare_stocktwits.py --prices-db data/database/experiment-b.sqlite
python -m xsentiment --config config/stocktwits.toml run --experiment D --db data/database/experiment-d.sqlite --out runs/reproduced-d
```

The first command retrieves public, bounded byte ranges from the publisher's labelled shards. The preparation step uses BTC.X labels and converts only complete groups of 24 historical hourly candles into daily bars. The conservative date-availability delay reflects the missing intraday timestamp. The resulting 17-origin holdout provides limited statistical precision.

## Storage and audit

SQLAlchemy Core defines raw_tweets, normalized_posts, processed_tweets, price_bars, news_articles, sentiment_features, model_features, predictions, evaluation_metrics, ingestion_runs and experiments. SQLite-specific upserts live in the repository module. Each experiment has a version tied to its inputs, source code, configuration and library versions. The database contains both OOS and latest deployment predictions; forecast.json identifies the actual fit time, observed cutoff and selected specification.

The local Windows environment is recorded in requirements-lock.txt. The portable package dependencies and optional financial-model extra are in pyproject.toml. Full local run manifests verify input snapshots, fitted models, plots and CSVs. The submission manifest verifies the report subset shared on GitHub. [Submission checklist](docs/submission.md).
