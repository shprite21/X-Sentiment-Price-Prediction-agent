"""Inspect metadata and a bounded prefix of public Kaggle downloads."""
import csv
import io
import json
import struct
import zlib
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

SOURCES = {
    "recent": ("pokeash/bitcoin-tweets-dataset-20252026", "bitcoin_tweets_latest.csv"),
    "bitcoin_small": ("kaushiksuresh147/bitcoin-tweets", "Bitcoin_tweets_dataset_2.csv"),
    "bitcoin_main": ("kaushiksuresh147/bitcoin-tweets", "Bitcoin_tweets.csv"),
}


def inspect(pair):
    key, (slug, filename) = pair
    result = {"slug": slug, "filename": filename}
    base = Path("data/source_metadata")
    base.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(Request(f"https://www.kaggle.com/api/v1/datasets/view/{slug}", headers={"User-Agent": "xsentiment-research/0.1"}), timeout=30) as response:
            metadata = json.load(response)
        (base / f"{key}_view.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        result["metadata"] = {k: v for k, v in metadata.items() if k in {"title", "subtitle", "description", "lastUpdated", "currentVersionNumber", "licenses"}}
        url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}/{filename}"
        with urlopen(Request(url, headers={"User-Agent": "xsentiment-research/0.1", "Range": "bytes=0-524287"}), timeout=60) as response:
            raw = response.read(524288)
            result["http"] = {"status": response.status, "type": response.headers.get("Content-Type"),
                "range": response.headers.get("Content-Range"), "length": response.headers.get("Content-Length"),
                "disposition": response.headers.get("Content-Disposition")}
        (base / f"{key}_prefix.bin").write_bytes(raw)
        if raw[:4] == b"PK\x03\x04":
            header = struct.unpack("<4s5H3I2H", raw[:30])
            method, name_len, extra_len = header[3], header[-2], header[-1]
            start = 30 + name_len + extra_len
            data = raw[start:]
            if method == 8:
                data = zlib.decompressobj(-15).decompress(data, max_length=4000000)
            elif method != 0:
                raise ValueError(f"Unsupported compression {method}")
        else:
            data = raw
        text = data.decode("utf-8-sig", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))[:5]
        result["columns"] = list(rows[0]) if rows else []
        result["sample_dates"] = [{k: v for k, v in row.items() if k and any(x in k.lower() for x in ["date", "time"])} for row in rows]
    except Exception as exc:
        result["error"] = str(exc)
    (base / f"{key}_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {key: result}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(inspect, SOURCES.items()):
            print(json.dumps(item, ensure_ascii=True), flush=True)
