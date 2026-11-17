import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from xsentiment.ingest import fetch_news, fetch_x, normalize_post, request_json
from xsentiment.storage import Store


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "test.sqlite")

    def test_apify_daily_budget_pagination_and_resume(self):
        run = {
            "id": "test-run",
            "status": "SUCCEEDED",
            "defaultDatasetId": "test-dataset",
            "finishedAt": "2026-01-02T00:00:00Z",
            "usageTotalUsd": 0.01,
        }
        post = {"id": "one", "text": "Bitcoin bullish", "createdAt": "2026-01-01T12:00:00Z"}
        from types import SimpleNamespace

        with (
            patch.dict(os.environ, {"APIFY_TOKEN": "test-secret"}, clear=True),
            patch("xsentiment.providers.ApifyClient") as sdk,
        ):
            client = sdk.return_value
            start = client.actor.return_value.start
            start.return_value = run
            client.run.return_value.wait_for_finish.return_value = run
            page = client.dataset.return_value.list_items
            page.side_effect = [SimpleNamespace(items=[post] * 1000), SimpleNamespace(items=[post])] * 2
            for _ in range(2):
                fetch_x(
                    self.store,
                    "2026-01-01",
                    "2026-01-02",
                    self.root / "raw",
                    budget_usd=1,
                    items_per_day=1001,
                )
            self.assertEqual(start.call_count, 1)
            self.assertEqual(page.call_args.kwargs["offset"], 1000)
            self.assertEqual(start.call_args.kwargs["max_total_charge_usd"], 1)
            self.assertEqual(start.call_args.kwargs["run_input"]["tweetLanguage"], "en")
            with self.assertRaisesRegex(ValueError, "Cached Apify"):
                fetch_x(
                    self.store,
                    "2026-01-01",
                    "2026-01-02",
                    self.root / "raw",
                    budget_usd=2,
                    items_per_day=1001,
                )
        self.assertEqual(len(self.store.frame("posts")), 1)
        for path in (self.root / "raw").glob("*.json"):
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))

    def test_tavily_news_preserves_publication_and_retrieval(self):
        raw = {
            "results": [
                {
                    "url": "https://example.com/news",
                    "title": "Bitcoin news",
                    "content": "Market context",
                    "published_date": "2026-01-01",
                }
            ]
        }
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-secret"}, clear=True),
            patch("xsentiment.providers.TavilyClient") as mock,
        ):
            mock.return_value.search.return_value = raw
            self.assertEqual(fetch_news(self.store, self.root / "raw"), 1)
        self.assertEqual(mock.return_value.search.call_args.kwargs["topic"], "news")
        row = self.store.frame("news").iloc[0]
        self.assertEqual(row.published_at, "2026-01-01T00:00:00+00:00")
        self.assertTrue(row.retrieved_at)

    def test_paid_post_is_not_retried_after_network_timeout(self):
        with patch("xsentiment.ingest.urlopen", side_effect=URLError("timeout")) as mock:
            with self.assertRaisesRegex(RuntimeError, "not retried"):
                request_json("https://example.com/actor", payload={"a": 1})
        self.assertEqual(mock.call_count, 1)

    def test_legacy_epoch_seconds_and_milliseconds(self):
        for stamp in [1767225600, "1767225600", "1767225600000"]:
            row = normalize_post({"timestamp": stamp, "text": "Bitcoin bullish"})
            self.assertEqual(row["created_at"], "2026-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
