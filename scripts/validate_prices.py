"""Record independent public-price ingestion evidence without requiring X/news keys."""
import json
from pathlib import Path
import pandas as pd
from xsentiment.storage import Store


root = Path(__file__).resolve().parents[1]
store = Store(root / "data/live.sqlite")
p = store.frame("prices").sort_values("time")
times = pd.to_datetime(p.time, utc=True)
expected = pd.date_range(times.min(), times.max(), freq="h")
missing = expected.difference(pd.DatetimeIndex(times))
assert len(p) == 4320, f"Expected 180 days x 24 bars, got {len(p)}"
assert len(missing) == 0
assert (p[["open", "high", "low", "close"]] > 0).all().all()
assert (p.high >= p[["open", "close", "low"]].max(axis=1)).all()
assert (p.low <= p[["open", "close", "high"]].min(axis=1)).all()
evidence = {"provider": "Coinbase Exchange public candles", "asset": "BTC-USD", "bar_seconds": 3600,
            "rows": len(p), "first_close_utc": times.min().isoformat(), "last_close_utc": times.max().isoformat(),
            "missing_hourly_bars": len(missing), "ohlc_checks": "passed", "source": "REAL_DATA",
            "requested_window": store.metadata().get("price_request"),
            "note": "Only public price ingestion verified. No live X/Tavily model evaluation is claimed."}
path = root / "deliverables/live-price-validation.json"
path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
