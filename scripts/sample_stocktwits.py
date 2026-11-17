"""Bounded, systematic byte-range sample from each publisher-listed labelled StockTwits shard."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://stocktwits-nyu.s3.us-west-2.amazonaws.com/"
PREFIX = "dataset/v1/data/csv/symbol_sentiments/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sample(root, slices=8, size=65536):
    manifest = ET.parse(urlopen(BASE + "?list-type=2&prefix=" + PREFIX, timeout=45))
    shards = [
        (n.findtext("s:Key", namespaces=NS), int(n.findtext("s:Size", namespaces=NS)))
        for n in manifest.findall("s:Contents", NS)
    ]
    root.mkdir(parents=True, exist_ok=True)
    columns = ["message_id", "user_id", "created_at", "sentiment", "symbol_list"]

    def read(shard):
        key, length = shard
        selected, spans = [], []
        for i in range(slices):
            start = i * (length - size) // slices
            end = start + size - 1
            path = root / (Path(key).stem + f"-{start}.bin")
            if not path.exists():
                req = Request(BASE + key, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(req, timeout=45) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{length}"
                    ):
                        raise ValueError("Server did not return requested range")
                    body = response.read(size + 1)
                    if len(body) != size:
                        raise ValueError("Incomplete range")
                    path.write_bytes(body)
            body = path.read_bytes()
            lines = body.splitlines()[1:-1]
            if i == 0 and body.splitlines()[0].decode().strip().split(",") != columns:
                raise ValueError("Unexpected source header")
            rows = csv.DictReader(io.StringIO(b"\n".join(lines).decode()), fieldnames=columns)
            for row in rows:
                if re.search(r"['\"]BTC\.X['\"]", row["symbol_list"] or ""):
                    selected.append(row)
            spans.append(
                dict(
                    key=key,
                    total_bytes=length,
                    first=start,
                    last=end,
                    sha256=hashlib.sha256(body).hexdigest(),
                )
            )
        return selected, spans

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pieces = list(pool.map(read, shards))
    rows = {r["message_id"]: r for group, _ in pieces for r in group}
    with (root / "btc-labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())
    spans = [span for _, group in pieces for span in group]
    (root / "ranges.json").write_text(json.dumps(spans, indent=2), encoding="utf-8")
    print(f"{len(shards)} shards, {len(spans) * size:,} bytes, {len(rows):,} unique BTC.X messages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/stocktwits"))
    sample(parser.parse_args().out)
