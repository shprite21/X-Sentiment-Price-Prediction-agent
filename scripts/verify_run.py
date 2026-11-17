"""Verify a run's artifact checksums without loading its model or running provider calls."""
import hashlib
import json
from pathlib import Path
import sys


def verify(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["artifact_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise SystemExit(f"Missing or changed artifacts: {failures}")
    print(f"Verified {len(manifest['artifact_sha256'])} artifacts; mode={manifest['mode']}")


if __name__ == "__main__":
    verify(sys.argv[1])
