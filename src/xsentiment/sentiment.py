"""Fixed text scorers. VADER outputs normalized lexical weights; FinBERT outputs softmax probabilities."""

import hashlib
import importlib.metadata
import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def analyzer():
    model = SentimentIntensityAnalyzer()
    model.lexicon.update(
        {"bullish": 2.0, "bearish": -2.0, "moon": 1.5, "rugpull": -3.0, "hodl": 1.0, "rekt": -2.0}
    )
    return model


def clean_text(text):
    text = re.sub(r"https?://\S+|@\w+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def score(text):
    scores = analyzer().polarity_scores(clean_text(text))
    return scores["pos"] - scores["neg"]


@lru_cache(maxsize=1)
def load_finbert(folder):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = AutoModelForSequenceClassification.from_pretrained(
        folder, local_files_only=True, use_safetensors=True
    )
    return AutoTokenizer.from_pretrained(folder, local_files_only=True), model.eval()


def score_posts(texts: pd.Series, cfg, batch_size=32) -> tuple[pd.DataFrame, str]:
    labels = ["positive", "negative", "neutral"]
    if cfg.sentiment_engine == "finbert":
        import torch

        tokenizer, model = load_finbert(cfg.finbert_path)
        indices = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        with (Path(cfg.finbert_path) / "model.safetensors").open("rb") as handle:
            version = "finbert:" + hashlib.file_digest(handle, "sha256").hexdigest()
        for name in ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"):
            version += ":" + hashlib.sha256((Path(cfg.finbert_path) / name).read_bytes()).hexdigest()[:16]
        cache = Path(cfg.finbert_path).parent / "finbert_scores.sqlite"
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        with closing(sqlite3.connect(cache, timeout=60)) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS scores (version TEXT, digest TEXT, pos REAL, neg REAL, neu REAL, PRIMARY KEY(version,digest))"
            )
            cached = {}
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" for _ in chunk)
                for key, *vals in con.execute(
                    f"SELECT digest,pos,neg,neu FROM scores WHERE version=? AND digest IN ({marks})",
                    [version, *chunk],
                ):
                    cached[key] = vals
            pending = {k: v for k, v in zip(keys, texts) if k not in cached}
            # Group similar lengths to reduce padding. Restore the original row order below.
            todo = sorted(pending, key=lambda k: len(pending[k]))
            with torch.inference_mode():
                for start in range(0, len(todo), batch_size):
                    chunk = todo[start : start + batch_size]
                    inputs = tokenizer(
                        [pending[k] for k in chunk],
                        truncation=True,
                        max_length=128,
                        padding=True,
                        return_tensors="pt",
                    )
                    probs = model(**inputs).logits.softmax(dim=-1)
                    vals = probs[:, [indices[label] for label in labels]].tolist()
                    with con:
                        con.executemany(
                            "INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?)",
                            [(version, k, *p) for k, p in zip(chunk, vals)],
                        )
                    cached.update(zip(chunk, vals))
            weights = [cached[k] for k in keys]
    else:
        weights = []
        for text in texts:
            result = analyzer().polarity_scores(clean_text(text))
            weights.append([result[k] for k in ("pos", "neg", "neu")])
        version = "vader-finance-v1:" + importlib.metadata.version("vaderSentiment")
    scores = pd.DataFrame(weights, columns=labels, index=texts.index)
    scores = scores.div(scores.sum(axis=1).replace(0, np.nan), axis=0).fillna(
        {"neutral": 1, "positive": 0, "negative": 0}
    )
    scores["sentiment"] = scores.positive - scores.negative
    if cfg.sentiment_engine == "finbert":
        scores["label"] = scores[labels].idxmax(axis=1)
    else:
        scores["label"] = np.select(
            [scores.sentiment > 0.05, scores.sentiment < -0.05], labels[:2], default="neutral"
        )
    return scores, version
