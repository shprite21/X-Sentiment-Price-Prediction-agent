"""Bounded-memory, deterministic hourly sampling from the timestamped Kaggle archive."""

import csv
import hashlib
import heapq
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from .config import utc
from .ingest import normalize_post, save_json


class HourlySampler:
    """Keep k smallest SHA-256 ranks per hour; independent of archive row order."""

    def __init__(self, cap=20, seed=42):
        if cap <= 0:
            raise ValueError("Hourly sample cap must be positive")
        self.cap, self.seed, self.buckets = cap, seed, {}

    def add(self, stamp, item):
        key = stamp[:13]
        identity = hashlib.sha256((stamp + "\0" + item["text"]).encode()).hexdigest()
        rank = int(hashlib.sha256((str(self.seed) + identity).encode()).hexdigest(), 16)
        heap, rows = self.buckets.setdefault(key, ([], {}))
        if identity in rows:
            return
        if len(heap) < self.cap:
            heapq.heappush(heap, (-rank, identity))
        elif rank < -heap[0][0]:
            _, removed = heapq.heapreplace(heap, (-rank, identity))
            rows.pop(removed)
        else:
            return
        rows[identity] = item

    def records(self):
        return [row for key in sorted(self.buckets) for _, row in sorted(self.buckets[key][1].items())]


def sample_archive(store, archive, *, start, end, posts_per_hour=20, seed=42):
    path = Path(archive)
    left, right = utc(start).to_pydatetime(), utc(end).to_pydatetime()
    prior = store.metadata().get("historical_sample")
    if prior and (prior["start"], prior["end"], prior["seed"], prior["sample_cap_per_hour"]) != (
        left.isoformat(),
        right.isoformat(),
        seed,
        posts_per_hour,
    ):
        raise ValueError("Choose a separate database for a different historical sampling cohort")
    if right <= left:
        raise ValueError("End must be after start")
    sampler = HourlySampler(posts_per_hour, seed)
    counts = {
        "rows_scanned": 0,
        "invalid_rows": 0,
        "outside_window": 0,
        "retweets_excluded": 0,
        "eligible_in_window": 0,
    }
    hourly_eligible = {}
    minimum, maximum = None, None
    csv.field_size_limit(2_000_000)
    with ZipFile(path) as archive_file:
        members = [n for n in archive_file.namelist() if Path(n).name == "Bitcoin_tweets.csv"]
        if len(members) != 1:
            raise ValueError("Expected exactly one Bitcoin_tweets.csv member")
        with (
            archive_file.open(members[0]) as binary,
            io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle,
        ):
            reader = csv.DictReader(handle)
            if not {"date", "text"}.issubset(reader.fieldnames or []):
                raise ValueError("Dataset must retain original date and text columns")
            for item in reader:
                counts["rows_scanned"] += 1
                if counts["rows_scanned"] % 500000 == 0:
                    print(json.dumps(counts), flush=True)
                try:
                    # Reject malformed CSV rows; never infer dates from text or account creation.
                    if None in item or not item.get("date") or not item.get("text"):
                        raise ValueError("Malformed row")
                    stamp = datetime.fromisoformat(item["date"])
                    stamp = (
                        stamp.replace(tzinfo=timezone.utc)
                        if stamp.tzinfo is None
                        else stamp.astimezone(timezone.utc)
                    )
                except (ValueError, TypeError):
                    counts["invalid_rows"] += 1
                    continue
                minimum = stamp if minimum is None else min(minimum, stamp)
                maximum = stamp if maximum is None else max(maximum, stamp)
                if not left <= stamp < right:
                    counts["outside_window"] += 1
                    continue
                if str(item.get("is_retweet", "")).lower() in {"true", "1"}:
                    counts["retweets_excluded"] += 1
                    continue
                counts["eligible_in_window"] += 1
                hour = stamp.isoformat()[:13]
                hourly_eligible[hour] = hourly_eligible.get(hour, 0) + 1
                sampler.add(stamp.isoformat(), item)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    verified = digest == "2ce71e16fd28191b18e0ead4430955868c1fba22d341f54190ea4a9255314fc3"
    source = "kaggle:kaushiksuresh147/bitcoin-tweets:v54" if verified else "local-archive:" + digest[:16]
    if prior and prior["archive_sha256"] != digest:
        raise ValueError("Archive differs from this database's historical cohort")
    records = sampler.records()
    acquired = datetime.now(timezone.utc).isoformat()
    store.save_raw(records, source, "X", acquired)
    selected = [
        normalize_post(
            row,
            source=source,
            observed_at=acquired,
            license="CC0-1.0" if verified else None,
            method="hourly hash sample",
        )
        for row in records
    ]
    selected = [r for r in selected if r]
    if not selected:
        raise ValueError("No usable timestamped posts in the requested interval")
    store.put("posts", selected)
    metadata = {
        "source": source,
        "source_url": "https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets"
        if verified
        else None,
        "version": 54 if verified else None,
        "archive": path.name,
        "archive_sha256": digest,
        "license": "CC0-1.0" if verified else None,
        "start": left.isoformat(),
        "end": right.isoformat(),
        "sample_cap_per_hour": posts_per_hour,
        "seed": seed,
        "sampling": "Smallest SHA-256 ranks of timestamp+text per UTC hour; no target or sentiment selection",
        "selected_posts": len(selected),
        "hours_with_eligible_posts": len(hourly_eligible),
        "source_min_date": minimum.isoformat(),
        "source_max_date": maximum.isoformat(),
        **counts,
        "counts_warning": "Model post volume is capped sampled volume; full eligible counts are provenance only",
        "language_warning": "Source has no language field; English sentiment scorer applied with multilingual limitation",
    }
    store.meta("mode", "REAL_HISTORICAL")
    store.meta("historical_sample", metadata)
    store.record(
        digest + left.isoformat() + right.isoformat() + f"/{seed}/{posts_per_hour}",
        source,
        "succeeded",
        metadata,
    )
    store.meta("source_hourly_eligible", hourly_eligible)
    save_json(path.parent / "historical_sample.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata
