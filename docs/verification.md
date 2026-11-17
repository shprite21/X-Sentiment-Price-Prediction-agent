# Execution record | 17 November 2026

Python 3.12.14 on Windows. The editable package with its FinBERT extra installed successfully. Pip's dependency consistency check passed.

| Executed check | Outcome |
|---|---|
| Full unittest suite | 29 passed, including future-post and future-price perturbations, per-fold scaler and imputer checks, holdout isolation, DST/weekends/early closes, SDK mocks, storage idempotence and replay checksums. |
| Historical reconstruction | Verified 702,724,074 compressed bytes, streamed 4,693,091 rows and restored all 4,320 completed Coinbase bars from 15 cached provider pages. |
| Historical X research | Actual local batched ProsusAI FinBERT inference, four models, five purged development folds plus the final chronological block. |
| StockTwits supplement | Downloaded 272 fixed ranges across 34 shards, filtered BTC.X, retained source hashes and ran a separate daily comparison. |
| Yahoo Finance | Live public request returned all 4,320 scheduled recent BTC-USD hourly bars. |
| Synthetic demo | Ran the complete 180-day pipeline without provider calls, including model fitting, charts, forecasts, database persistence and manifest. |
| Historical replay | Rebuilt from checksummed canonical input snapshots in a fresh database and compared the OOS predictions, incremental statistics and selected strategy path. |
| Credential gates | Missing --live refused external collection. Apify/Tavily/Exa SDK contracts exercised using fixtures. No paid run or retrieval occurred. |
| Research limitations | Historical data and FinBERT scores are retrospective. No validated recent X/news observations support Experiment A, Experiment C or a current directional sentiment call. |

Local full runs support `python scripts/verify_run.py reports/historical`. The published subset has its own hash inventory in reports/reproducibility_manifest.json. Source tweet text, model binaries, private data, SQLite files and transient outputs are excluded from Git.
