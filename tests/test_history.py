import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from zipfile import ZipFile

from xsentiment.history import HourlySampler, sample_archive
from xsentiment.storage import Store


class HistoricalSamplingTests(unittest.TestCase):
    def test_sampling_is_order_independent_and_capped_per_hour(self):
        records = [
            (f"2021-02-10T{hour:02d}:{i:02d}:00+00:00", {"text": f"Bitcoin {hour} {i}"})
            for hour in [1, 2]
            for i in range(50)
        ]
        first, second = HourlySampler(5), HourlySampler(5)
        for stamp, row in records:
            first.add(stamp, row)
        for stamp, row in reversed(records):
            second.add(stamp, row)
        self.assertEqual(first.records(), second.records())
        self.assertEqual(len(first.records()), 10)

    def test_duplicate_does_not_consume_sample_slot(self):
        sampler = HourlySampler(5)
        sampler.add("2021-02-10T01:00:00+00:00", {"text": "Bitcoin bullish", "likes": "1"})
        sampler.add("2021-02-10T01:00:00+00:00", {"text": "Bitcoin bullish", "likes": "2"})
        self.assertEqual(len(sampler.records()), 1)

    def test_archive_dates_are_preserved_and_retweets_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=["date", "text", "is_retweet", "user_followers"])
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "date": "2021-02-10 01:30:00",
                        "text": "Bitcoin bullish",
                        "is_retweet": "False",
                        "user_followers": "50",
                    },
                    {"date": "2021-02-10 01:45:00", "text": "Bitcoin bearish", "is_retweet": "True"},
                    {"date": "2026-02-10 01:30:00", "text": "outside window"},
                ]
            )
            archive = root / "test.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("Bitcoin_tweets.csv", buffer.getvalue())
            store = Store(root / "test.sqlite")
            with redirect_stdout(io.StringIO()):
                metadata = sample_archive(store, archive, start="2021-02-10", end="2021-02-11")
            posts = store.frame("posts")
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts.created_at.iloc[0], "2021-02-10T01:30:00+00:00")
            self.assertEqual(posts.followers.iloc[0], 50)
            self.assertEqual(metadata["outside_window"], 1)
            self.assertEqual(metadata["retweets_excluded"], 1)
            self.assertEqual(store.metadata()["mode"], "REAL_HISTORICAL")


if __name__ == "__main__":
    unittest.main()
