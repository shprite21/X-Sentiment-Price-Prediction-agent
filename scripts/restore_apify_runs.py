"""Restore the published experiment's Apify datasets using read-only SDK methods."""

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

from xsentiment.config import Config, utc
from xsentiment.ingest import normalize_post, save_json
from xsentiment.storage import Store


def restore(manifest_path, store, raw_dir, cost_audit=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = Config(**manifest["config"])
    store.bind(cfg)
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    collection = manifest["apify"]
    billing = []
    for row in collection["runs"]:
        remote = client.run(row["id"]).get()
        if not remote or remote["status"] != "SUCCEEDED" or remote["defaultDatasetId"] != row["dataset_id"]:
            raise ValueError("Recorded run or dataset unavailable to this account")
        stamp = utc(remote["finishedAt"]).isoformat()
        prices = (remote.get("pricingInfo") or {}).get("pricingPerEvent", {}).get("actorChargeEvents", {})
        counts = remote.get("chargedEventCounts") or {}
        known = all(name in prices and prices[name].get("eventPriceUsd") is not None for name in counts)
        event_usd = (
            sum((Decimal(str(prices[name]["eventPriceUsd"])) * n for name, n in counts.items()), Decimal(0))
            if known and counts
            else None
        )
        billing.append(
            {
                "run_id": row["id"],
                "finished_at": stamp,
                "status": remote["status"],
                "configured_cap_usd": row["run_budget_usd"],
                "usage_total_usd": remote.get("usageTotalUsd"),
                "charged_events": counts,
                "event_prices_usd": {n: v.get("eventPriceUsd") for n, v in prices.items()},
                "charged_events_usd": float(event_usd) if event_usd is not None else None,
            }
        )
        dataset = client.dataset(row["dataset_id"])
        accepted = 0
        for offset in range(0, row["raw"], 1000):
            page = dataset.list_items(
                offset=offset, limit=min(1000, row["raw"] - offset), clean=False
            ).items
            save_json(raw_dir / f"{row['id']}-{offset}.json", page)
            store.save_raw(page, "apify", "X", stamp)
            normalized = [normalize_post(item, "apify", stamp, method="Apify") for item in page]
            valid = [
                r for r in normalized if r and utc(row["start"]) <= utc(r["created_at"]) < utc(row["end"])
            ]
            accepted += len(valid)
            store.put("posts", valid)
        if accepted != row["posts"]:
            raise ValueError(f"Provider dataset contents changed for {row['id']}")
        store.record(row["id"], "apify", "restored", row)
    store.meta("x_collection", collection)
    if cost_audit:
        save_json(cost_audit, {"actor_id": collection["actor_id"], "runs": billing})
    print(f"Restored {len(collection['runs'])} recorded Apify runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/recent_x/manifest.json"))
    parser.add_argument("--db", default="data/database/restored-a.sqlite")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/apify-restored"))
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--cost-audit", type=Path, help="Write sanitized provider billing metadata")
    args = parser.parse_args()
    load_dotenv()
    restore(args.manifest, Store(args.db), args.raw, args.cost_audit)
