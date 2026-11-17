"""Provider calls use stdlib HTTP, retries for reads, and bounded actor spending."""

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import utc
from .providers import fetch_news as fetch_news
from .providers import fetch_x as fetch_x


def request_json(url, *, payload=None, headers=None, retries=4):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(
        url,
        data=body,
        headers={
            "User-Agent": "xsentiment-research/0.1",
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    # Never automatically repeat a POST: a timeout can mean a paid actor started.
    attempts = retries if payload is None else 1
    for attempt in range(attempts):
        try:
            with urlopen(req, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(
                    f"Provider HTTP {exc.code}; verify credentials, quota and input"
                ) from None
            time.sleep(min(30, float(exc.headers.get("Retry-After", 2**attempt))))
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise RuntimeError(
                    "Provider network error; POSTs are not retried to avoid duplicate runs"
                ) from None
            time.sleep(2**attempt)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, ensure_ascii=False, allow_nan=False)
    Path(tmp.name).replace(path)


def fetch_prices(store, start, end, raw_dir, *, cached=False):
    """Coinbase timestamps are bucket OPEN times; the database records CLOSE times."""
    start, end = utc(start).floor("h"), utc(end).floor("h")
    cursor = start
    all_rows = []
    while cursor < end:
        stop = min(cursor + pd.Timedelta(hours=299), end)
        params = urlencode({"start": cursor.isoformat(), "end": stop.isoformat(), "granularity": 3600})
        path = Path(raw_dir) / f"coinbase_{cursor.strftime('%Y%m%d%H')}.json"
        if cached:
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = request_json(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?{params}")
            save_json(path, raw)
        for item in raw:
            if not isinstance(item, list) or len(item) != 6:
                raise ValueError("Unexpected Coinbase candle schema")
            stamp, low, high, opened, closed, volume = item
            bar_open = pd.Timestamp(stamp, unit="s", tz="UTC")
            if start <= bar_open < end:
                all_rows.append(
                    {
                        "time": (bar_open + pd.Timedelta(hours=1)).isoformat(),
                        "open": opened,
                        "high": high,
                        "low": low,
                        "close": closed,
                        "volume": volume,
                        "source": "coinbase",
                    }
                )
        cursor = stop
        print(f"Prices through {stop.isoformat()}", flush=True)
        if not cached:
            time.sleep(0.35)
    if not all_rows:
        raise ValueError("Provider returned no prices for the requested date range")
    if len({r["time"] for r in all_rows}) != (end - start) // pd.Timedelta(hours=1):
        raise ValueError("Incomplete Coinbase candle window")
    store.put("prices", all_rows)
    details = {"start": start.isoformat(), "end": end.isoformat(), "provider": "coinbase", "cached": cached}
    store.meta("price_request", details)
    store.record("coinbase/" + start.isoformat() + "/" + end.isoformat(), "coinbase", "succeeded", details)
    return len({r["time"] for r in all_rows})


def normalize_post(item, source="apify", observed_at=None, *, platform="X", license=None, method=None):
    text = item.get("text") or item.get("full_text") or item.get("content")
    stamp = item.get("createdAt") or item.get("created_at") or item.get("date") or item.get("timestamp")
    if (
        not isinstance(text, str)
        or not text.strip()
        or stamp is None
        or str(item.get("isRetweet", item.get("is_retweet", ""))).lower() in {"true", "1"}
    ):
        return None
    author = item.get("author") or {}
    if not isinstance(author, dict):
        author = {}

    def number(*values):
        for value in values:
            if value is not None and str(value).strip():
                try:
                    return max(0, int(float(str(value).replace(",", ""))))
                except (ValueError, TypeError, OverflowError):
                    continue
        return None

    original_timestamp = str(stamp)
    if re.fullmatch(r"\d{10}(?:\.\d+)?", str(stamp)):
        stamp = pd.Timestamp(float(stamp), unit="s", tz="UTC")
    elif re.fullmatch(r"\d{13}", str(stamp)):
        stamp = pd.Timestamp(int(stamp), unit="ms", tz="UTC")
    else:
        try:
            stamp = utc(stamp)
        except (ValueError, TypeError, OverflowError):
            return None
    if pd.isna(stamp):
        return None
    original_id = item.get("id_str") or item.get("id")
    ident = original_id or hashlib.sha256((stamp.isoformat() + "\0" + text).encode()).hexdigest()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    seen = utc(observed_at).isoformat() if observed_at is not None else now
    available = item.get("available_at")
    return {
        "id": str(ident),
        "created_at": stamp.isoformat(),
        "text": str(text),
        "likes": number(item.get("likeCount"), item.get("likes")),
        "reposts": number(item.get("retweetCount"), item.get("reposts"), item.get("retweets")),
        "views": number(item.get("viewCount"), item.get("views")),
        "followers": number(
            author.get("followers"), author.get("followersCount"), item.get("user_followers")
        ),
        "source": source,
        "observed_at": seen,
        "original_id": str(original_id) if original_id else None,
        "original_timestamp": original_timestamp,
        "platform": platform,
        "processed_at": now,
        "method": method or ("apify" if source == "apify" else "file import"),
        "license": license,
        "available_at": utc(available).isoformat() if available else None,
        "username": author.get("userName") or item.get("username"),
        "author_name": author.get("name") or item.get("user_name"),
        "author_id": author.get("id") or item.get("author_id") or item.get("user_id"),
        "post_type": "reply"
        if item.get("inReplyToId") or item.get("isReply") is True
        else "original"
        if "isRetweet" in item or "is_retweet" in item
        else "unknown",
    }


def import_posts(
    store, path, *, start=None, end=None, chunk_size=2000, dataset=None, license=None, platform="X"
):
    """Stream CSV/JSONL histories; filter to the matching price window before storing."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        handle = path.open(encoding="utf-8-sig", newline="")
        reader = csv.DictReader(handle)
    elif path.suffix.lower() == ".jsonl":
        handle = path.open(encoding="utf-8")
        reader = (json.loads(line) for line in handle if line.strip())
    elif path.suffix.lower() == ".json":
        handle = path.open(encoding="utf-8")
        reader = iter(json.load(handle))
    else:
        raise ValueError("Use CSV, JSONL or an Apify JSON array")
    counts = {"read": 0, "accepted": 0, "rejected": 0, "outside_window": 0}
    buffer, raw = [], []
    dataset = dataset or f"import:{path.name}"
    acquired = pd.Timestamp.now(tz="UTC").isoformat()
    try:
        for item in reader:
            counts["read"] += 1
            if not isinstance(item, dict):
                counts["rejected"] += 1
                continue
            raw.append(item)
            if len(raw) >= chunk_size:
                store.save_raw(raw, dataset, platform, acquired)
                raw = []
            try:
                row = normalize_post(
                    item, source=dataset, observed_at=acquired, platform=platform, license=license
                )
            except (ValueError, TypeError):
                row = None
            if row is None:
                counts["rejected"] += 1
                continue
            stamp = utc(row["created_at"])
            if (start is not None and stamp < utc(start)) or (end is not None and stamp >= utc(end)):
                counts["outside_window"] += 1
                continue
            buffer.append(row)
            counts["accepted"] += 1
            if len(buffer) >= chunk_size:
                store.put("posts", buffer)
                buffer = []
        store.save_raw(raw, dataset, platform, acquired)
        store.put("posts", buffer)
    finally:
        handle.close()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    details = {"file": path.name, "sha256": digest, "platform": platform, "license": license, **counts}
    store.meta("historical_import", details)
    store.record(dataset + "/" + digest, dataset, "succeeded" if counts["accepted"] else "empty", details)
    if not counts["accepted"]:
        raise ValueError(f"No usable posts matched the requested window: {counts}")
    return counts
