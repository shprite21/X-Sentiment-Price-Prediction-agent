"""Prepare a separate daily BTC.X label experiment using aligned historical Coinbase candles."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from xsentiment.config import Config
from xsentiment.storage import Store, timestamp


def prepare(root, source_db, db):
    cfg = Config(platform="StockTwits", interval="1d")
    store = Store(db)
    store.bind(cfg)
    path = root / "btc-labels.csv"
    labels = pd.read_csv(path, dtype={"message_id": str, "user_id": str})
    original = Store(source_db).frame("prices").set_index("time").sort_index()
    original.index = pd.to_datetime(original.index, utc=True)
    if not original.index.equals(pd.date_range(original.index.min(), original.index.max(), freq="h")):
        raise ValueError("Hourly source has gaps")
    grouped = original.resample("1D", closed="right", label="right")
    p = grouped.agg(dict(open="first", high="max", low="min", close="last"))
    p = p.loc[grouped.size().eq(24)].copy()
    p["time"], p["source"] = p.index.map(pd.Timestamp.isoformat), "coinbase_daily_from_hourly"
    store.put("prices", p.to_dict("records"))
    acquired = timestamp()
    source = "stocktwits-symbol-labels"
    store.save_raw(labels.to_dict("records"), source, "StockTwits", acquired)
    rows = []
    for row in labels.itertuples():
        day = pd.Timestamp(row.created_at, tz="UTC")
        # Date precision only. By 48h after UTC midnight the local source day has ended everywhere.
        rows.append(
            dict(
                id=row.message_id,
                original_id=row.message_id,
                author_id=row.user_id,
                original_timestamp=row.created_at,
                created_at=day.isoformat(),
                available_at=(day + pd.Timedelta(days=2)).isoformat(),
                platform="StockTwits",
                source=source,
                observed_at=acquired,
                processed_at=acquired,
                native_label=row.sentiment,
                method="public S3 fixed-range sample",
                license="see publisher",
            )
        )
    store.put("posts", rows)
    details = dict(
        provider="NYU StockTwits public S3",
        unique_btc_labels=len(rows),
        ranges_sha256=hashlib.sha256((root / "ranges.json").read_bytes()).hexdigest(),
        sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    store.meta("stocktwits_sample", details)
    store.record(details["sample_sha256"], source, "succeeded", details)
    print(f"{len(rows):,} BTC.X labels, {len(p)} complete daily OHLC bars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/stocktwits"))
    ap.add_argument("--prices-db", default="data/database/experiment-b.sqlite")
    ap.add_argument("--db", default="data/database/experiment-d.sqlite")
    args = ap.parse_args()
    prepare(args.raw, args.prices_db, args.db)
