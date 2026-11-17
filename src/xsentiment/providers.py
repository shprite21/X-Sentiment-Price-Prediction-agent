"""Official SDK adapters. Only the CLI's explicit live mode starts provider requests."""

import hashlib
import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from apify_client import ApifyClient
from exa_py import Exa
from tavily import TavilyClient

from .calendar import bars
from .config import Config, utc

LOG = logging.getLogger(__name__)
ACTOR = "apidojo~tweet-scraper"
DEFAULT_TEMPLATE = {
    "searchTerms": ["{query}"],
    "start": "{start}",
    "end": "{end}",
    "sort": "Latest",
    "tweetLanguage": "en",
    "maxItems": "{limit}",
}


def actor_input(template, cfg, left, right, limit):
    terms = dict.fromkeys([cfg.ticker, "$" + cfg.ticker, cfg.company, *cfg.keywords])
    lookup = {
        "{query}": "("
        + " OR ".join('"' + s + '"' if " " in s else s for s in terms)
        + ") -filter:retweets",
        "{ticker}": cfg.ticker,
        "{company}": cfg.company,
        "{start}": left.isoformat(),
        "{end}": right.isoformat(),
        "{limit}": limit,
    }

    def render(v):
        if isinstance(v, dict):
            if any(
                any(k in key.lower() for k in ("token", "password", "secret", "cookie", "apikey"))
                for key in v
            ):
                raise ValueError("Use a credential-free search input template")
            return {k: render(val) for k, val in v.items()}
        if isinstance(v, list):
            return [render(val) for val in v]
        return lookup.get(v, v)

    return render(template)


def fetch_x(store, start, end, raw_dir, *, items_per_day=300, budget_usd=5.0, cfg=None):
    from .ingest import normalize_post, save_json

    cfg = cfg or Config()
    token = os.environ.get("APIFY_TOKEN")
    if not token or min(budget_usd, items_per_day) <= 0:
        raise ValueError("Configure APIFY_TOKEN and positive budget and result limits")
    actor = (os.getenv("APIFY_ACTOR_ID") or ACTOR).replace("/", "~")
    template_file = os.getenv("APIFY_INPUT_FILE")
    if not template_file and actor != ACTOR:
        raise ValueError("Supply APIFY_INPUT_FILE matching the selected Actor's published schema")
    template = (
        json.loads(Path(template_file).read_text(encoding="utf-8-sig"))
        if template_file
        else DEFAULT_TEMPLATE
    )
    client = ApifyClient(token, max_retries=0)
    left, end = utc(start), utc(end)
    intervals = []
    while left < end:
        right = min(left.normalize() + pd.Timedelta(days=1), end)
        intervals.append((left, right))
        left = right
    runs = []
    for left, right in intervals:
        inputs = actor_input(template, cfg, left, right, items_per_day)
        plan = dict(actor=actor, input=inputs, budget=budget_usd / len(intervals))
        identity = hashlib.sha256(f"{cfg.asset}/{actor}/{left}/{right}".encode()).hexdigest()[:24]
        prefix = Path(raw_dir) / ("apify_" + identity)
        plan_path, run_file = prefix.with_suffix(".request.json"), prefix.with_suffix(".run.json")
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Cached Apify request differs from this input or budget")
        if plan_path.exists() and not run_file.exists():
            raise ValueError(
                "Previous start has unknown status. Reconcile the run ID in Apify Console before resuming"
            )
        record = dict(
            actor=actor, input=inputs, asset=cfg.asset, start=left.isoformat(), end=right.isoformat()
        )
        try:
            if run_file.exists():
                run = json.loads(run_file.read_text())
            else:
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation claims a paid start across concurrent collectors.
                with plan_path.open("x", encoding="utf-8") as handle:
                    json.dump(plan, handle)
                run = client.actor(actor).start(
                    run_input=inputs,
                    max_items=items_per_day,
                    max_total_charge_usd=Decimal(str(plan["budget"])),
                )
                run = {k: str(run[k]) for k in ("id", "status", "defaultDatasetId") if k in run}
                save_json(run_file, run)
            record["id"] = run["id"]
            final = client.run(run["id"]).wait_for_finish(wait_secs=1800)
            if not final or final["status"] != "SUCCEEDED":
                raise RuntimeError("Incomplete Apify run")
            dataset = client.dataset(final["defaultDatasetId"])
            offset, accepted = 0, 0
            acquired = utc(final["finishedAt"]).isoformat()
            while offset < items_per_day:
                limit = min(1000, items_per_day - offset)
                page = dataset.list_items(offset=offset, limit=limit, clean=False).items
                save_json(prefix.with_suffix(f".page-{offset}.json"), page)
                store.save_raw(page, "apify", "X", acquired)
                rows = [normalize_post(item, "apify", acquired, method="Apify") for item in page]
                valid = [r for r in rows if r and left <= utc(r["created_at"]) < right]
                store.put("posts", valid)
                accepted += len(valid)
                offset += len(page)
                if len(page) < limit:
                    break
            record.update(
                dataset_id=final["defaultDatasetId"],
                build=final.get("buildId"),
                raw=offset,
                posts=accepted,
                cap_hit=offset >= items_per_day,
            )
            runs.append(record)
            store.record(identity, "apify", "succeeded", record)
        except Exception as exc:
            store.record(identity, "apify", "incomplete", dict(record, error=type(exc).__name__))
            LOG.error("Apify collection incomplete window=%s run=%s", left, record.get("id"))
            raise RuntimeError(
                f"Apify collection incomplete ({type(exc).__name__}); inspect ingestion_runs"
            ) from None
        store.meta("x_collection", dict(actor_id=actor, budget_cap_usd=budget_usd, runs=runs))
    return runs


def fetch_news(store, raw_dir, cfg=None):
    from .ingest import save_json

    cfg = cfg or Config()
    now = pd.Timestamp.now(tz="UTC")
    query = f"{cfg.company} {cfg.ticker} announcements earnings and market developments"
    providers = ["tavily"] + (["exa"] if os.getenv("EXA_API_KEY") else [])
    total = 0
    for provider in providers:
        key = provider + "-" + now.strftime("%Y%m%dT%H%M%S%f")
        try:
            if provider == "tavily":
                raw = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
                    query=query, topic="news", time_range="week", max_results=10, search_depth="basic"
                )
                save_json(Path(raw_dir) / (key + ".json"), raw)
                items = raw.get("results", [])
            else:
                response = Exa(api_key=os.environ["EXA_API_KEY"]).search(
                    query,
                    category="news",
                    num_results=10,
                    start_published_date=(now - pd.Timedelta(days=7)).isoformat(),
                    contents={"highlights": True},
                )
                items = [
                    dict(
                        url=r.url,
                        title=r.title,
                        published_date=r.published_date,
                        content=" ".join(r.highlights or []),
                    )
                    for r in response.results
                ]
                save_json(Path(raw_dir) / (key + ".json"), items)
            rows = []
            for item in items:
                u = urlsplit(item.get("url") or "")
                if u.scheme not in {"http", "https"} or not u.hostname:
                    continue
                date = pd.to_datetime(item.get("published_date"), errors="coerce", utc=True)
                rows.append(
                    dict(
                        url=urlunsplit((u.scheme, u.netloc.lower(), u.path, u.query, "")),
                        title=item.get("title"),
                        content=item.get("content"),
                        publisher=u.hostname,
                        source=provider,
                        published_at=date.isoformat() if pd.notna(date) else None,
                        retrieved_at=now.isoformat(),
                    )
                )
            store.put("news", rows)
            store.record(key, provider, "succeeded", {"items": len(rows), "query": query})
            total += len(rows)
        except Exception as exc:
            store.record(key, provider, "incomplete", {"error": type(exc).__name__})
            raise RuntimeError(f"{provider} search failed ({type(exc).__name__})") from None
    store.meta("news_collection", {"providers": providers, "query": query, "retrieved_at": now.isoformat()})
    return total


def fetch_yahoo(store, start, end, raw_dir, cfg):
    """Map Yahoo's opening labels to session closes, retaining split-adjusted OHLC and Adj Close."""
    import yfinance as yf

    from .ingest import save_json

    left, right = utc(start), utc(end)
    schedule = bars(left, right, cfg)
    schedule = schedule.loc[(schedule.open >= left) & (schedule.close <= right)]
    if schedule.empty:
        raise ValueError("No completed sessions in window")
    mapping = dict(zip(schedule.open, schedule.close))
    days = {row.open.tz_convert(cfg.timezone).date(): row.close for row in schedule.itertuples()}
    folder = Path(raw_dir) / "yahoo"
    folder.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f"{cfg.asset}/{left}/{right}/{cfg.interval}".encode()).hexdigest()[:16]
    try:
        history = yf.Ticker(cfg.asset).history(
            start=left.to_pydatetime(),
            end=right.to_pydatetime(),
            interval=cfg.interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
            raise_errors=True,
        )
        history.to_csv(folder / (tag + ".csv"))
        rows = []
        for stamp, row in history.iterrows():
            close = mapping.get(utc(stamp)) if cfg.interval == "1h" else days.get(stamp.date())
            if close is None:
                continue
            rows.append(
                dict(
                    time=close.isoformat(),
                    open=row.Open,
                    high=row.High,
                    low=row.Low,
                    close=row.Close,
                    adj_close=row.get("Adj Close"),
                    volume=row.get("Volume"),
                    source="yfinance",
                )
            )
        if len({r["time"] for r in rows}) != len(schedule):
            raise ValueError(f"Received {len(rows)} of {len(schedule)} scheduled Yahoo bars")
        store.put("prices", rows)
        details = dict(
            asset=cfg.asset,
            provider="yfinance",
            start=left.isoformat(),
            end=right.isoformat(),
            bars=len(rows),
        )
        store.record(tag, "yfinance", "succeeded", details)
        store.meta("price_request", details)
        save_json(folder / (tag + ".json"), details)
        return len(rows)
    except Exception as exc:
        store.record(
            tag,
            "yfinance",
            "incomplete",
            dict(start=left.isoformat(), end=right.isoformat(), error=type(exc).__name__),
        )
        raise RuntimeError(
            f"Yahoo history unavailable for requested window ({type(exc).__name__}: {exc})"
        ) from None
