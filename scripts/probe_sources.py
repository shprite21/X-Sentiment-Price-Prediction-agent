"""Small public metadata probes; no dataset archives are downloaded."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


URLS = {
    "recent_kaggle": "https://www.kaggle.com/api/v1/datasets/list/pokeash/bitcoin-tweets-dataset-20252026",
    "bitcoin_kaggle": "https://www.kaggle.com/api/v1/datasets/list/kaushiksuresh147/bitcoin-tweets",
    "legacy_kaggle": "https://www.kaggle.com/api/v1/datasets/list/alaix14/bitcoin-tweets-20160101-to-20190329",
    "hf_schema": "https://datasets-server.huggingface.co/first-rows?dataset=StephanAkkerman%2Fcrypto-stock-tweets&config=default&split=train",
}


def probe(pair):
    name, url = pair
    try:
        with urlopen(Request(url, headers={"User-Agent": "xsentiment-research/0.1"}), timeout=30) as response:
            body = response.read(1500000)
            content_type = response.headers.get("Content-Type", "")
        try:
            value = json.loads(body)
        except ValueError:
            return name, {"url": url, "content_type": content_type, "error": "non-JSON response", "prefix": body[:120].decode(errors="replace")}
        dest = Path("data/source_metadata")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
        if name == "hf_schema":
            summary = {"features": value.get("features"), "sample_rows": value.get("rows", [])[:1]}
        else:
            summary = value
        return name, {"url": url, "response": summary}
    except HTTPError as exc:
        return name, {"url": url, "http_status": exc.code}
    except Exception as exc:
        return name, {"url": url, "error": str(exc)}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, result in pool.map(probe, URLS.items()):
            print(json.dumps({name: result}, ensure_ascii=True), flush=True)
