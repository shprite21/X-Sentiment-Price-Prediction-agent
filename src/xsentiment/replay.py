"""Restore canonical run inputs without provider access or deserializing model files."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import Config


def restore_inputs(directory, store):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if any(not store.frame(table).empty for table in ["prices", "posts", "news"]):
        raise ValueError("Replay requires a fresh empty database")
    frames = {}
    for table in ["prices", "posts", "news"]:
        path = root / f"input_{table}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["inputs"][table]["sha256"]:
            raise ValueError(f"Input checksum mismatch: {table}")
        df = pd.read_csv(
            path,
            dtype={c: str for c in ("id", "original_id", "author_id", "username", "author_name", "text")}
            if table == "posts"
            else None,
            keep_default_na=False,
            na_values=[""],
            float_precision="round_trip",
        )
        frames[table] = df.astype(object).where(pd.notna(df), None)
    cfg = Config(**manifest["config"])
    for table, frame in frames.items():
        store.put(table, frame.to_dict(orient="records"))
    for key, value in manifest.get("collection_metadata", {}).items():
        store.meta(key, value)
    store.meta("replayed_from", {"run_mode": manifest["mode"], "created_at": manifest["created_at"]})
    return cfg
