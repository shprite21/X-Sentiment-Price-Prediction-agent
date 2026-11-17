"""Compare dates only for matching public tweet text in two bounded source prefixes."""
import csv
import io
import json
import struct
import zlib
from datetime import datetime
from pathlib import Path


def rows(path):
    data = path.read_bytes()
    if data[:4] == b"PK\x03\x04":
        header = struct.unpack("<4s5H3I2H", data[:30])
        data = zlib.decompressobj(-15).decompress(data[30 + header[-2] + header[-1]:], 4000000)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace"), newline=""))
    result = []
    for _, row in zip(range(200), reader):
        if row.get("date") and row.get("text"):
            result.append(row)
    return result


base = Path("data/source_metadata")
recent = rows(base / "recent_prefix.bin")
original = {r["text"]: r for r in rows(base / "bitcoin_main_prefix.bin")}
matching = [(r, original[r["text"]]) for r in recent if r["text"] in original]
result = {"recent_sample_rows": len(recent), "matching_text_rows_in_original_prefix": len(matching),
    "different_dates_for_matching_text": sum(r["date"] != o["date"] for r, o in matching),
    "exact_five_year_shifts": sum(datetime.fromisoformat(o["date"]).replace(year=2026).isoformat(sep=" ") == r["date"] for r, o in matching),
    "date_examples_only": [{"recent_date": r["date"], "original_date": o["date"]} for r, o in matching[:5]],
    "decision": "Exclude the 2025-2026 repack from time-aligned evaluation pending timestamp provenance verification"}
(base / "timestamp_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
