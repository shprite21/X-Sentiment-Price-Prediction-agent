import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from xsentiment.calendar import bars, future_closes
from xsentiment.config import Config, utc
from xsentiment.demo import populate
from xsentiment.features import build_features
from xsentiment.ingest import import_posts, normalize_post
from xsentiment.model import evaluate, fit_model, select_model
from xsentiment.providers import fetch_news, fetch_yahoo
from xsentiment.sentiment import score_posts
from xsentiment.storage import Store


class IntegrityTests(unittest.TestCase):
    def test_fold_preprocessing_and_selection_isolation(self):
        idx = pd.date_range("2022-01-01", periods=155, freq="h", tz="UTC")
        signal = np.arange(len(idx), dtype=float)
        signal[::3] = np.nan
        frame = pd.DataFrame(
            dict(
                close=100 + np.arange(len(idx)),
                open=100,
                price_signal=signal,
                social=np.sin(np.arange(len(idx))),
            ),
            index=idx,
        )
        config = Config(min_train=40, test_size=23, holdout=25)
        checked = []

        def inspect_fit(x, y, alpha, name, seed):
            m = fit_model(x, y, alpha, name, seed)
            if name in ("price_only", "sentiment_price"):
                np.testing.assert_allclose(m[0].statistics_, x.median().to_numpy())
                np.testing.assert_allclose(m[1].mean_, m[0].transform(x).mean(axis=0))
                checked.append(len(x))
            return m

        with patch("xsentiment.model.fit_model", side_effect=inspect_fit):
            preds, folds = evaluate(frame, ["price_signal"], ["price_signal", "social"], config)
        self.assertGreater(len(set(checked)), 1)
        self.assertLess(utc(folds[-2]["last_test_label"]), utc(folds[-1]["test_start"]))
        first = select_model(preds)
        preds.loc[preds.phase == "holdout", "actual_return"] = 100000
        self.assertEqual(select_model(preds), first)

    def test_delayed_post_missingness_and_price_indicators(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "db")
            cfg = Config(days=8)
            populate(store, cfg)
            p, s = store.frame("prices"), store.frame("posts")
            t = utc(p.time.iloc[100])
            row = s.iloc[[0]].copy()
            row["created_at"] = (t - pd.Timedelta(minutes=2)).isoformat()
            row["available_at"] = (t + pd.Timedelta(minutes=2)).isoformat()
            row["observed_at"] = (t + pd.Timedelta(hours=2, minutes=2)).isoformat()
            f, pc, ac, _ = build_features(p, row, cfg)
            self.assertTrue(np.isnan(f.loc[t, "sentiment"]))
            self.assertEqual(f.loc[t, "has_posts"], 0)
            self.assertEqual(f.loc[t + pd.Timedelta(hours=1), "post_count"], 1)
            strict, *_ = build_features(p, row, replace(cfg, availability="first_seen"))
            self.assertEqual(strict.loc[t + pd.Timedelta(hours=3), "post_count"], 1)
            self.assertAlmostEqual(f.loc[t, "return_5h"], np.log(p.close.iloc[100] / p.close.iloc[95]))
            self.assertAlmostEqual(f.loc[t, "high_6"], p.close.iloc[100] / p.high.iloc[95:101].max() - 1)
            self.assertFalse({"volume", "likes", "news", "close", "target_time"} & set(ac))

    def test_calendar_early_close_dst_and_weekend(self):
        cfg = Config(
            asset="AAPL", ticker="AAPL", company="Apple", exchange="XNYS", timezone="America/New_York"
        )
        grid = bars("2024-11-29", "2024-11-30", cfg)
        friday = grid[grid.open.dt.date == utc("2024-11-29").date()]
        self.assertEqual(friday.close.iloc[-1], utc("2024-11-29T18:00Z"))
        self.assertEqual(len(friday), 4)
        nxt = future_closes(utc("2024-03-08T21:00Z"), cfg)[0]
        self.assertEqual(nxt, utc("2024-03-11T14:30Z"))
        daily = replace(cfg, interval="1d")
        self.assertEqual(future_closes(utc("2024-03-08T21:00Z"), daily)[0], utc("2024-03-11T20:00Z"))

    def test_scores_neutral_and_sign(self):
        vals, version = score_posts(
            pd.Series(["bullish gains strong", "bearish losses weak", "Bitcoin table"]), Config()
        )
        np.testing.assert_allclose(vals[["positive", "negative", "neutral"]].sum(axis=1), 1)
        self.assertGreater(vals.sentiment.iloc[0], 0)
        self.assertLess(vals.sentiment.iloc[1], 0)
        self.assertEqual(vals.label.iloc[2], "neutral")
        self.assertIn("vader", version)
        self.assertIsNone(normalize_post({"date": "invalid", "text": "BTC"}))
        self.assertIsNone(normalize_post({"date": "2025-01-01", "text": "BTC"})["likes"])

    def test_yahoo_calendar_and_archive(self):
        cfg = Config(
            asset="AAPL",
            ticker="AAPL",
            company="Apple",
            interval="1d",
            exchange="XNYS",
            timezone="America/New_York",
        )
        idx = pd.DatetimeIndex(["2024-11-29"], tz=cfg.timezone)
        frame = pd.DataFrame(
            {
                "Open": [100],
                "Close": [102],
                "High": [104],
                "Low": [99],
                "Adj Close": [101],
                "Volume": [100],
            },
            index=idx,
        )
        with tempfile.TemporaryDirectory() as tmp, patch("yfinance.Ticker") as ticker:
            store = Store(Path(tmp) / "db")
            ticker.return_value.history.return_value = frame
            for _ in range(2):
                fetch_yahoo(store, "2024-11-29", "2024-11-30", Path(tmp), cfg)
            rows = store.frame("prices")
            self.assertEqual(len(rows), 1)
            self.assertEqual(utc(rows.time.iloc[0]), utc("2024-11-29T18Z"))
            self.assertEqual(rows.adj_close.iloc[0], 101)
            self.assertFalse(ticker.return_value.history.call_args.kwargs["auto_adjust"])
            self.assertTrue(list((Path(tmp) / "yahoo").glob("*.csv")))

    def test_raw_and_cross_source_post_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "db")
            path = Path(tmp) / "tweets.csv"
            path.write_text(
                "id,date,text\n42,2026-01-01,Bitcoin bullish\n43,bad,Bitcoin\n", encoding="utf-8"
            )
            for source in ["one", "one", "two"]:
                import_posts(store, path, dataset=source, license="CC0-1.0")
            self.assertEqual(len(store.frame("posts")), 2)
            with store.connect() as c:
                self.assertEqual(c.execute("SELECT count(*) FROM raw_tweets").fetchone()[0], 4)
                self.assertEqual(c.execute("SELECT count(*) FROM ingestion_runs").fetchone()[0], 2)

    def test_exa_optional_sdk_and_news_dedupe(self):
        from types import SimpleNamespace

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("os.environ", {"TAVILY_API_KEY": "fixture", "EXA_API_KEY": "fixture"}),
            patch("xsentiment.providers.TavilyClient") as tv,
            patch("xsentiment.providers.Exa") as ex,
        ):
            store = Store(Path(tmp) / "db")
            tv.return_value.search.return_value = {
                "results": [{"title": "BTC", "url": "https://example.org/a#top"}]
            }
            ex.return_value.search.return_value = SimpleNamespace(
                results=[
                    SimpleNamespace(
                        title="BTC",
                        url="https://example.org/a",
                        published_date="2026-01-01",
                        highlights=["Example"],
                    )
                ]
            )
            fetch_news(store, Path(tmp))
            self.assertEqual(len(store.frame("news")), 1)
            self.assertEqual(store.metadata()["news_collection"]["providers"], ["tavily", "exa"])
