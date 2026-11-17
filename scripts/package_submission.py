"""Package the same tracked files as the reviewed Git submission."""

import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
destination = root / "deliverables" / "x-sentiment-assessment.zip"
destination.parent.mkdir(exist_ok=True)
if not any(p.startswith("reports/figures/") for p in paths):
    raise SystemExit("Stage the reviewed source and reports first")
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name in filter(None, paths):
        archive.write(root / name, arcname=name)
print(f"Packaged {len(paths) - 1} reviewed files into {destination.name}")
