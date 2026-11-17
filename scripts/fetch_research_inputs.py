"""Prepare pinned public research inputs. Providers that need paid credentials are separate."""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

ARCHIVE_HASH = "2ce71e16fd28191b18e0ead4430955868c1fba22d341f54190ea4a9255314fc3"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/kaushiksuresh147/bitcoin-tweets/Bitcoin_tweets.csv?datasetVersionNumber=54"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch():
    path = Path("data/downloads/bitcoin-tweets-v54.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".part")
        with (
            urlopen(
                Request(ARCHIVE_URL, headers={"User-Agent": "xsentiment-research"}), timeout=90
            ) as response,
            partial.open("wb") as dest,
        ):
            shutil.copyfileobj(response, dest, length=1024 * 1024)
        if checksum(partial) != ARCHIVE_HASH:
            raise ValueError(
                "Unexpected archive bytes; verify the version and any login requirement on Kaggle"
            )
        partial.replace(path)
    if checksum(path) != ARCHIVE_HASH:
        raise ValueError("Local archive checksum differs from the verified single-member ZIP")
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    snapshot_download(
        "ProsusAI/finbert",
        revision=REVISION,
        local_dir="data/models/finbert",
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
    )
    if (
        checksum(Path("data/models/finbert/model.safetensors"))
        != "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00"
    ):
        raise ValueError("FinBERT weights checksum mismatch")
    print("Verified historical archive and FinBERT weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.parse_args()
    fetch()
