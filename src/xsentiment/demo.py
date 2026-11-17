"""Deterministic fixture data. These are NOT historical BTC prices or X posts."""

import numpy as np
import pandas as pd


def populate(store, cfg):
    if not store.frame("prices").empty or not store.frame("posts").empty:
        raise ValueError("Demo requires a fresh database to prevent mixing synthetic and real data")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.days * 24
    times = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    latent = np.zeros(n)
    for i in range(1, n):
        latent[i] = 0.6 * latent[i - 1] + rng.normal(0, 0.5)
    # Weak synthetic relationship, explicitly disclosed, never evidence of alpha.
    returns = rng.normal(0, 0.003, n) + 0.00015 * np.r_[0, latent[:-1]]
    closes = 70000 * np.exp(np.cumsum(returns))
    opens = np.r_[70000, closes[:-1]]
    spread = rng.uniform(0.0001, 0.004, n)
    prices = [
        {
            "time": t.isoformat(),
            "open": float(opens[i]),
            "close": float(closes[i]),
            "high": float(max(opens[i], closes[i]) * (1 + spread[i])),
            "low": float(min(opens[i], closes[i]) * (1 - spread[i])),
            "volume": float(rng.uniform(10, 200)),
            "source": "SYNTHETIC_DEMO",
        }
        for i, t in enumerate(times)
    ]
    posts = []
    for i, t in enumerate(times):
        for j in range(rng.poisson(3)):
            bullish = rng.random() < 1 / (1 + np.exp(-latent[i]))
            phrase = "bullish optimistic strong gains" if bullish else "bearish worried weak losses"
            posts.append(
                {
                    "id": f"demo-{i}-{j}",
                    "created_at": (t - pd.Timedelta(minutes=int(rng.integers(1, 60)))).isoformat(),
                    "text": f"Bitcoin $BTC {phrase}. Research fixture {i}:{j}",
                    "likes": int(rng.integers(0, 100)),
                    "reposts": int(rng.integers(0, 25)),
                    "views": int(rng.integers(0, 1000)),
                    "followers": int(rng.integers(0, 10000)),
                    "source": "SYNTHETIC_DEMO",
                    "observed_at": t.isoformat(),
                }
            )
    store.put("prices", prices)
    store.put("posts", posts)
    store.put(
        "news",
        [
            {
                "url": "https://example.com/synthetic-fixture",
                "title": "Synthetic news fixture: no current market claim",
                "content": "Demo only. Live news requires a Tavily key.",
                "published_at": times[-1].isoformat(),
                "retrieved_at": times[-1].isoformat(),
                "source": "SYNTHETIC_DEMO",
            }
        ],
    )
    store.meta("mode", "SYNTHETIC_DEMO")
    store.meta("seed", cfg.seed)
