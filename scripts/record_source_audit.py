"""Export source-verification evidence without republishing full tweet text."""
import hashlib
import json
from pathlib import Path


base = Path("data/source_metadata")
comparison = json.loads((base / "timestamp_comparison.json").read_text(encoding="utf-8"))
hf = json.loads((base / "hf_schema.json").read_text(encoding="utf-8"))
result = {"checked_on": "2026-11-17", "timestamp_comparison": comparison,
          "compared_sources": ["https://www.kaggle.com/datasets/pokeash/bitcoin-tweets-dataset-20252026",
                               "https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets"],
          "method": "Download bounded file prefixes; compare exact tweet text and original date fields. Non-random prefix audit, not a whole-dataset authenticity claim.",
          "prefix_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                            for name in ["recent_prefix.bin", "bitcoin_main_prefix.bin"]},
          "hf_source": "https://huggingface.co/datasets/StephanAkkerman/crypto-stock-tweets",
          "hf_columns": [f["name"] for f in hf["features"]],
          "hf_decision": "No timestamp column; unsuitable for a time-aligned price backtest without returning to source data",
          "chosen_source": "kaushiksuresh147/bitcoin-tweets version 54",
          "historical_backtest": json.loads(Path("data/downloads/historical_sample.json").read_text(encoding="utf-8"))}
Path("deliverables/source_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Recorded source audit and checksums")
