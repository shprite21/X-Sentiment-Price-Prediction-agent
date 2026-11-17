"""Command line entry points for collection, offline reconstruction, evaluation and replay."""

import argparse
import json
import logging
import os
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import Config, utc
from .demo import populate
from .history import sample_archive
from .ingest import fetch_news, fetch_prices, fetch_x, import_posts
from .pipeline import run
from .providers import ACTOR, fetch_yahoo
from .replay import restore_inputs
from .storage import Store


def main(argv=None):
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s level=%(levelname)s module=%(name)s %(message)s"
    )
    for name in ("httpx", "httpcore", "apify_client", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Multi-horizon X and price research")
    parser.add_argument("--config", type=Path, help="TOML file containing a [research] section")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo", help="End-to-end synthetic offline verification")
    demo.add_argument("--db", default="data/database/demo.sqlite")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--days", type=int, default=180)
    ingest = sub.add_parser(
        "ingest", help="Explicit live provider calls or replay of cached Coinbase candles"
    )
    ingest.add_argument("--only", choices=["prices", "x", "news", "all"], default="all")
    ingest.add_argument("--provider", choices=["yahoo", "coinbase"], default="yahoo")
    ingest.add_argument("--live", action="store_true")
    ingest.add_argument("--cached", action="store_true", help="Read existing Coinbase JSON pages")
    ingest.add_argument("--raw", default="data/raw")
    ingest.add_argument("--db", default="data/database/recent.sqlite")
    ingest.add_argument("--start")
    ingest.add_argument("--end")
    ingest.add_argument("--days", type=int)
    ingest.add_argument("--items-per-day", "--items-per-run", type=int, default=300)
    ingest.add_argument("--window-days", type=int, default=1)
    ingest.add_argument("--query-days", type=int, default=2)
    ingest.add_argument("--budget-usd", type=float, default=5.0)
    imp = sub.add_parser("import-posts", help="Stream an asset-specific CSV/JSONL or read a JSON array")
    imp.add_argument("path", type=Path)
    imp.add_argument("--dataset")
    imp.add_argument("--license")
    imp.add_argument("--start")
    imp.add_argument("--end")
    imp.add_argument("--db", default="data/database/history.sqlite")
    history = sub.add_parser("prepare-history", help="Stream the original timestamped Bitcoin v54 archive")
    history.add_argument("archive")
    history.add_argument("--db", default="data/database/experiment-b.sqlite")
    history.add_argument("--start", default="2021-02-06")
    history.add_argument("--end", default="2021-08-05")
    history.add_argument("--posts-per-hour", type=int, default=20)
    train = sub.add_parser(
        "run", aliases=["run-all"], help="Build features, train, evaluate, predict, plot and report"
    )
    train.add_argument("--db", default="data/database/recent.sqlite")
    train.add_argument("--out", default="runs/recent")
    train.add_argument("--experiment", choices=["A", "B", "C", "D"])
    train.add_argument("--dataset", action="append", help="Exact source name; may repeat")
    train.add_argument("--cost-bps", type=float)
    replay = sub.add_parser("replay", help="Offline run from checksummed input snapshots")
    replay.add_argument("directory")
    replay.add_argument("--db", default="data/database/replay.sqlite")
    replay.add_argument("--out", default="runs/replay")
    current = sub.add_parser(
        "current", help="Apply the selected historical specification at a recent cutoff"
    )
    current.add_argument("--research", default="reports/historical")
    current.add_argument("--db", default="data/database/recent.sqlite")
    current.add_argument("--out", default="runs/current")
    args = parser.parse_args(argv)
    cfg = (
        Config(**tomllib.loads(args.config.read_text(encoding="utf-8-sig"))["research"])
        if args.config
        else Config()
    )
    try:
        if args.command == "doctor":
            print(
                json.dumps(
                    dict(
                        python=sys.version.split()[0],
                        actor=os.getenv("APIFY_ACTOR_ID") or ACTOR,
                        credentials={
                            k: bool(os.getenv(k)) for k in ("APIFY_TOKEN", "TAVILY_API_KEY", "EXA_API_KEY")
                        },
                        local_finbert=(Path(cfg.finbert_path) / "model.safetensors").is_file(),
                    ),
                    indent=2,
                )
            )
            return 0
        store = Store(args.db)
        if args.command == "replay":
            original = json.loads((Path(args.directory) / "manifest.json").read_text(encoding="utf-8"))
            cfg = restore_inputs(args.directory, store)
            result = run(store, args.out, cfg, experiment=original.get("experiment"))
        elif args.command == "demo":
            cfg = replace(cfg, days=args.days, sentiment_engine="vader")
            if cfg.asset != "BTC-USD" or cfg.exchange != "24/7" or cfg.interval != "1h":
                raise ValueError("Seeded fixture uses hourly BTC-shaped data")
            store.bind(cfg)
            populate(store, cfg)
            result = run(store, args.out, cfg)
        elif args.command in {"run", "run-all"}:
            if args.cost_bps is not None:
                cfg = replace(
                    cfg, cost_bps=args.cost_bps, slippage_bps=min(args.cost_bps, cfg.slippage_bps)
                )
            result = run(store, args.out, cfg, experiment=args.experiment, datasets=args.dataset)
        elif args.command == "current":
            from .current import write_current

            write_current(Path(args.research), store, Path(args.out))
            return 0
        else:
            store.bind(cfg)
            if args.command == "prepare-history":
                if cfg.asset != "BTC-USD" or cfg.platform != "X":
                    raise ValueError("This archive contains Bitcoin tweets")
                sample_archive(
                    store,
                    args.archive,
                    start=args.start,
                    end=args.end,
                    posts_per_hour=args.posts_per_hour,
                    seed=cfg.seed,
                )
            elif args.command == "import-posts":
                print(
                    import_posts(
                        store,
                        args.path,
                        start=args.start,
                        end=args.end,
                        dataset=args.dataset,
                        license=args.license,
                        platform=cfg.platform,
                    )
                )
            elif args.command == "ingest":
                if args.cached:
                    if args.live or args.only != "prices" or args.provider != "coinbase":
                        raise ValueError(
                            "Cached ingestion requires --only prices --provider coinbase without --live"
                        )
                elif not args.live:
                    raise ValueError("External collection requires --live")
                now = pd.Timestamp.now(tz="UTC")
                end = utc(args.end or cfg.end) if args.end or cfg.end else now.floor("h")
                start = (
                    utc(args.start or cfg.start)
                    if args.start or cfg.start
                    else end - pd.Timedelta(days=args.days or cfg.days)
                )
                if args.only == "prices" and not (args.start or args.days or cfg.start):
                    existing = store.frame("posts")
                    if not existing.empty:
                        start = min(
                            start,
                            pd.to_datetime(existing.created_at, utc=True, format="mixed").min().floor("D"),
                        )
                if not start < end <= now:
                    raise ValueError("Choose a nonempty completed historical window")
                if args.only in ("x", "all") and not os.getenv("APIFY_TOKEN"):
                    raise ValueError("Set APIFY_TOKEN for live X collection")
                if args.only in ("news", "all") and not os.getenv("TAVILY_API_KEY"):
                    raise ValueError("Set TAVILY_API_KEY for primary news collection")
                if args.only in ("prices", "all"):
                    if args.provider == "yahoo":
                        print(fetch_yahoo(store, start, end, args.raw, cfg))
                    else:
                        if (cfg.asset, cfg.interval, cfg.exchange) != ("BTC-USD", "1h", "24/7"):
                            raise ValueError(
                                "Coinbase adapter requires hourly BTC-USD; use Yahoo for other assets"
                            )
                        print(fetch_prices(store, start, end, args.raw, cached=args.cached))
                if args.only in ("x", "all"):
                    fetch_x(
                        store,
                        start,
                        end,
                        args.raw,
                        cfg=cfg,
                        items_per_day=args.items_per_day,
                        window_days=args.window_days,
                        query_days=args.query_days,
                        budget_usd=args.budget_usd,
                    )
                if args.only in ("news", "all"):
                    fetch_news(store, args.raw, cfg)
            return 0
        print(
            json.dumps(
                {
                    "experiment": result["experiment"],
                    "selected_model": result["selected_model"],
                    "mode": result["mode"],
                    "output": args.out,
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
