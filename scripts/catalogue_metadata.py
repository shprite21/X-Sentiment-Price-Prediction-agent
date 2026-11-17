"""Inspect Kaggle metadata and the first 1 KiB of one listed file per candidate, without login."""

import concurrent.futures
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

SLUGS = [
    "omermetinn/tweets-about-the-top-companies-from-2015-to-2020",
    "alaix14/bitcoin-tweets-20160101-to-20190329",
    "andreapenasmartinez/bitcoin-twitter-sentiment-dataset-20132023",
    "kaushiksuresh147/bitcoin-tweets",
    "pokeash/bitcoin-tweets-dataset-20252026",
]
BASE = "https://www.kaggle.com/api/v1/datasets/"


def inspect(slug):
    result = {"slug": slug}
    try:
        with urlopen(BASE + "view/" + slug, timeout=25) as resp:
            view = json.load(resp)
        result.update(
            license=view.get("licenseName"),
            updated=view.get("lastUpdated"),
            version=view.get("currentVersionNumber"),
        )
        with urlopen(BASE + "list/" + slug + "?pageSize=100", timeout=25) as resp:
            listing = json.load(resp)
        result["files"] = {f["name"]: f["totalBytes"] for f in listing["datasetFiles"]}
        name = next(iter(result["files"]))
        req = Request(BASE + "download/" + slug + "/" + quote(name), headers={"Range": "bytes=0-1023"})
        with urlopen(req, timeout=45) as resp:
            chunk = resp.read(1024)
            result.update(
                file_probed=name,
                http_status=resp.status,
                received_bytes=len(chunk),
                content_type=resp.headers.get("Content-Type"),
                content_range=resp.headers.get("Content-Range"),
            )
    except Exception as exc:
        result["error"] = type(exc).__name__ + " " + str(getattr(exc, "code", ""))
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(inspect, SLUGS))
    path = Path("data/source_metadata/catalogue_probe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in result:
        print(
            row["slug"],
            row.get("license"),
            sum(row.get("files", {}).values()),
            row.get("http_status"),
            row.get("error"),
        )
