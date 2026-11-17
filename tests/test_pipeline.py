"""Tests focus on temporal integrity, financial arithmetic, and provider contracts."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from xsentiment.config import Config, utc
from xsentiment.demo import populate
from xsentiment.features import build_features, make_targets
from xsentiment.ingest import fetch_prices, import_posts, normalize_post
from xsentiment.model import evaluate, forecast, risk_metrics, splits, strategy_returns
from xsentiment.pipeline import run
from xsentiment.replay import restore_inputs
from xsentiment.storage import Store


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "test.sqlite")
        self.cfg = replace(Config(), days=12, min_train=72, test_size=48, holdout=48)
        populate(self.store, self.cfg)
        self.prices, self.posts = self.store.frame("prices"), self.store.frame("posts")
        self.frame, self.price_cols, self.all_cols, self.audit = build_features(self.prices, self.posts)

    def test_future_changes_do_not_change_past_features(self):
        cutoff = self.frame.index[90]
        changed_prices = self.prices.copy()
        future = pd.to_datetime(changed_prices.time, utc=True) > cutoff
        changed_prices.loc[future, ["open", "high", "low", "close"]] *= 1.5
        changed_posts = self.posts.copy()
        future_posts = pd.to_datetime(changed_posts.created_at, utc=True) >= cutoff
        changed_posts.loc[future_posts, "text"] = "catastrophic bearish future text"
        changed, *_ = build_features(changed_prices, changed_posts)
        pd.testing.assert_frame_equal(self.frame.loc[:cutoff], changed.loc[:cutoff])

    def test_exact_boundary_post_enters_next_hour(self):
        cutoff = self.frame.index[90]
        row = {
            **self.posts.iloc[0].to_dict(),
            "id": "boundary",
            "text": "Bitcoin unique bullish boundary test",
            "created_at": cutoff.isoformat(),
        }
        changed, *_ = build_features(
            self.prices, pd.concat([self.posts, pd.DataFrame([row])], ignore_index=True)
        )
        self.assertEqual(changed.loc[cutoff, "post_count"], self.frame.loc[cutoff, "post_count"])
        next_hour = cutoff + pd.Timedelta(hours=1)
        self.assertEqual(changed.loc[next_hour, "post_count"], self.frame.loc[next_hour, "post_count"] + 1)

    def test_engagement_cannot_affect_features(self):
        posts = self.posts.copy()
        posts[["likes", "reposts", "followers", "views"]] = 999999999
        changed, *_ = build_features(self.prices, posts)
        pd.testing.assert_frame_equal(changed, self.frame)

    def test_missing_price_hour_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            build_features(self.prices.drop(index=100), self.posts)

    def test_no_overlap_rejected(self):
        posts = self.posts.copy()
        posts["created_at"] = "2020-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_features(self.prices, posts)

    def test_targets_are_exact_horizon_log_returns(self):
        targets = make_targets(self.frame)
        for h in range(1, 6):
            self.assertAlmostEqual(
                targets.iloc[0][h], np.log(self.frame.close.iloc[h] / self.frame.close.iloc[0])
            )
        self.assertTrue(targets.iloc[-1].isna().all())

    def test_purge_and_nonoverlapping_test_origins(self):
        all_test = []
        for phase, train_idx, test_idx in splits(len(self.frame) - 5, self.cfg):
            self.assertLess(train_idx[-1] + 5, test_idx[0])
            all_test.extend(test_idx)
        self.assertEqual(len(all_test), len(set(all_test)))
        holdout_start = len(self.frame) - 5 - self.cfg.holdout
        self.assertEqual(
            all_test,
            list(range(all_test[0], holdout_start - 5)) + list(range(holdout_start, len(self.frame) - 5)),
        )

    def test_future_holdout_labels_do_not_change_predictions(self):
        baseline, folds = evaluate(self.frame, self.price_cols, self.all_cols, self.cfg)
        holdout = pd.Timestamp(folds[-1]["test_start"])
        changed = self.frame.copy()
        changed.loc[changed.index >= holdout, "close"] *= 2
        # Test feature vectors are unchanged; only target prices change.
        altered, _ = evaluate(changed, self.price_cols, self.all_cols, self.cfg)
        a = baseline[baseline.phase == "holdout"].predicted_return.to_numpy()
        b = altered[altered.phase == "holdout"].predicted_return.to_numpy()
        np.testing.assert_allclose(a, b)

    def test_risk_drawdown_includes_initial_capital(self):
        metrics = risk_metrics([-0.1, 0.0])
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["total_return"], -0.1)
        self.assertIsNone(risk_metrics([0, 0])["sharpe"])
        self.assertIsNone(risk_metrics([0.1, 0.2])["sortino"])

    def test_strategy_delay_costs_and_nonoverlapping_returns(self):
        rows = [
            {
                "time": t,
                "target_time": t + pd.Timedelta(hours=1),
                "horizon": 1,
                "threshold": 0.005,
                "predicted_return": 0.01,
                "actual_price": 102,
                "target_open": 100,
                "fold": 1,
            }
            for t in pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        ]
        curve = strategy_returns(pd.DataFrame(rows), 10)
        self.assertEqual(curve.time.iloc[0], utc("2026-01-01T01:00:00Z"))
        np.testing.assert_allclose(curve.gross_return, 0.02)
        np.testing.assert_allclose(curve.net_return, 1.02 * 0.999**2 - 1)
        np.testing.assert_allclose(curve.turnover, [2, 2, 2])

    def test_coinbase_open_timestamp_becomes_close(self):
        empty = Store(self.root / "prices.sqlite")
        raw = [[1767225600, 99, 102, 100, 101, 7]]  # 2026-01-01 00:00 UTC
        with (
            patch("xsentiment.ingest.request_json", return_value=raw),
            patch("xsentiment.ingest.time.sleep"),
        ):
            fetch_prices(empty, "2026-01-01", "2026-01-01T01:00Z", self.root / "raw")
        self.assertEqual(utc(empty.frame("prices").time.iloc[0]), utc("2026-01-01T01:00Z"))

    def test_normalizer_and_idempotent_import(self):
        row = {
            "id": "external-1",
            "createdAt": "Wed Jan 07 12:00:00 +0000 2026",
            "text": "Bitcoin bullish",
            "author": {"followers": 123},
            "likeCount": 10,
        }
        normalized = normalize_post(row)
        self.assertEqual(normalized["followers"], 123)
        self.assertEqual(normalized["likes"], 10)
        self.assertIsNone(normalize_post({**row, "isRetweet": True}))
        path = self.root / "import.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        before = len(self.store.frame("posts"))
        import_posts(self.store, path)
        import_posts(self.store, path)
        self.assertEqual(len(self.store.frame("posts")), before + 1)

    def test_new_york_dst_conversion(self):
        self.assertEqual(
            utc("2026-01-15").tz_convert("America/New_York").utcoffset().total_seconds(), -18000
        )
        self.assertEqual(
            utc("2026-07-15").tz_convert("America/New_York").utcoffset().total_seconds(), -14400
        )

    def test_forecast_has_observed_zero_and_five_future_prices(self):
        _, result = forecast(self.frame, self.all_cols, self.cfg)
        self.assertEqual([r["horizon"] for r in result], list(range(6)))
        self.assertEqual(result[0]["price"], self.frame.close.iloc[-1])
        self.assertEqual(result[0]["kind"], "observed")
        self.assertTrue(all(r["price"] > 0 for r in result))

    def test_end_to_end_outputs_and_synthetic_labels(self):
        with redirect_stdout(StringIO()):
            manifest = run(self.store, self.root / "out", self.cfg)
        self.assertEqual(manifest["mode"], "SYNTHETIC_DEMO")
        for name in [
            "evaluation.md",
            "dashboard.html",
            "sentiment_outlook.md",
            "forecast.json",
            "manifest.json",
        ]:
            self.assertTrue((self.root / "out" / name).exists())
        self.assertIn("SYNTHETIC DEMO", (self.root / "out" / "evaluation.md").read_text(encoding="utf-8"))
        self.assertNotIn("likes", manifest["features"])
        with self.store.connect() as con:
            self.assertGreater(con.execute("SELECT COUNT(*) FROM hourly_features").fetchone()[0], 0)
        replay_store = Store(self.root / "replay.sqlite")
        cfg = restore_inputs(self.root / "out", replay_store)
        self.assertEqual(cfg, self.cfg)
        restored, *_ = build_features(replay_store.frame("prices"), replay_store.frame("posts"))
        np.testing.assert_allclose(
            restored[self.all_cols], self.frame[self.all_cols], rtol=1e-12, atol=1e-12
        )
        bad = self.root / "out" / "input_prices.csv"
        bad.write_text(bad.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            restore_inputs(self.root / "out", Store(self.root / "tampered.sqlite"))


if __name__ == "__main__":
    unittest.main()
