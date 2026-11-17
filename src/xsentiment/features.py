"""Align observed information to exchange bar closes. Targets live in a separate frame."""

import re

import numpy as np
import pandas as pd

from .calendar import bars
from .config import Config
from .sentiment import clean_text, score_posts


def prepare_posts(posts, boundaries, cfg):
    x = posts.copy()
    x["created_at"] = pd.to_datetime(x.created_at, utc=True, errors="coerce", format="mixed")
    if "platform" in x:
        x = x.loc[x.platform == cfg.platform].copy()
    x["effective"] = x.created_at
    if "available_at" in x:
        available = pd.to_datetime(x.available_at, utc=True, errors="coerce", format="mixed")
        x["effective"] = x.effective.where(available.isna() | (x.effective >= available), available)
    if cfg.availability == "first_seen":
        observed = pd.to_datetime(x.observed_at, utc=True, errors="coerce", format="mixed")
        x = x.loc[observed.notna()].copy()
        x["effective"] = x.effective.where(x.effective >= observed.loc[x.index], observed.loc[x.index])
    x = x.loc[
        x.created_at.notna()
        & x.effective.ge(boundaries.open.iloc[0])
        & x.effective.lt(boundaries.close.iloc[-1])
    ].copy()
    x = x.sort_values(["effective", "created_at", "id"], kind="stable")
    duplicate_ids = int(x.id.duplicated().sum())
    x = x.drop_duplicates("id")
    x["clean"] = x.text.fillna("").map(clean_text)
    terms = sorted({cfg.company, cfg.ticker, *cfg.keywords} - {""})
    relevant = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)"
    usable = x.clean.str.contains(relevant, case=False, regex=True) & x.clean.str.contains(r"\w")
    native = cfg.platform == "StockTwits" and "native_label" in x and x.native_label.notna().all()
    if native:
        usable = x.native_label.isin([-1, 1])
    irrelevant = int((~usable).sum())
    x = x.loc[usable].copy()
    duplicate_text = 0 if native else int(x.clean.str.casefold().duplicated().sum())
    if not native:
        x = x.loc[~x.clean.str.casefold().duplicated()].copy()
    if x.empty:
        raise ValueError("No usable asset posts overlap the price window and availability policy")
    if native:
        engine = "StockTwits self-declared binary label"
        x["sentiment"] = x.native_label.astype(float)
        x["label"] = np.where(x.sentiment > 0, "positive", "negative")
        x["positive"], x["negative"], x["neutral"] = (
            (x.sentiment > 0).astype(float),
            (x.sentiment < 0).astype(float),
            0.0,
        )
    else:
        scores, engine = score_posts(x.clean, cfg)
        x = x.join(scores)
    x["time"] = pd.DatetimeIndex(boundaries.close)[boundaries.close.searchsorted(x.effective, side="right")]
    x["author"] = pd.Series(pd.NA, index=x.index, dtype="object")
    for column in ("author_id", "username", "author_name"):
        if column in x:
            x["author"] = x.author.fillna(x[column])
    audit = {
        "input_posts": len(posts),
        "duplicate_ids": duplicate_ids,
        "duplicate_text": duplicate_text,
        "irrelevant_or_empty": irrelevant,
        "unique_text_posts_in_window": len(x),
        "engine": engine,
        "first_post_utc": x.created_at.min().isoformat(),
        "last_post_utc": x.created_at.max().isoformat(),
    }
    return x, audit


def build_features(prices, posts, cfg=None, *, with_posts=False, price_only=False):
    cfg = cfg or Config()
    if prices.empty or (posts.empty and not price_only):
        raise ValueError("Both prices and posts are required")
    p = prices.copy()
    p["time"] = pd.to_datetime(p.time, utc=True, format="mixed")
    p = p.set_index("time").sort_index()
    grid = bars(p.index.min() - pd.Timedelta(days=4), p.index.max(), cfg)
    grid = grid.loc[grid.close.between(p.index.min(), p.index.max())].reset_index(drop=True)
    missing = pd.DatetimeIndex(grid.close).difference(p.index)
    if len(missing) or not p.index.equals(pd.DatetimeIndex(grid.close)):
        raise ValueError(f"Missing {len(missing)} scheduled bars or duplicate/off-calendar observations")
    p = p[["open", "high", "low", "close"]].astype(float)
    if (
        not np.isfinite(p).all().all()
        or (p <= 0).any().any()
        or (p.high < p.max(axis=1)).any()
        or (p.low > p.min(axis=1)).any()
    ):
        raise ValueError("Invalid OHLC observations")
    if price_only:
        x = pd.DataFrame(columns=["time", "sentiment", "author", "label"]).astype({"sentiment": float})
        x["time"] = pd.to_datetime(x.time, utc=True)
        audit = {"engine": "none", "input_posts": 0, "unique_text_posts_in_window": 0}
    else:
        x, audit = prepare_posts(posts, grid, cfg)
    grouped = x.groupby("time")
    agg = grouped.sentiment.agg(
        post_count="size",
        sentiment="mean",
        median_sentiment="median",
        sentiment_std=lambda v: v.std(ddof=0),
    )
    agg["unique_authors"] = grouped.author.nunique().where(grouped.author.count() > 0)
    for label, column in [
        ("positive", "bull_fraction"),
        ("negative", "bear_fraction"),
        ("neutral", "neutral_fraction"),
    ]:
        agg[column] = x.label.eq(label).groupby(x.time).mean()
    f = p.join(agg)
    f["post_count"] = f.post_count.fillna(0).astype(int)
    f["has_posts"] = f.post_count.gt(0).astype(int)
    f["log_post_count"] = np.log1p(f.post_count)
    f["log_authors"] = np.log1p(f.unique_authors)
    f["imbalance"] = f.bull_fraction - f.bear_fraction
    f["sentiment_change"] = f.sentiment.diff()
    f["sentiment_lag1"] = f.sentiment.shift(1)
    f["sentiment_lag3"] = f.sentiment.shift(3)
    f["log_volume_change"] = f.log_post_count.diff()
    log_close = np.log(f.close)
    ret = log_close.diff()
    price_cols = []
    suffix = "h" if cfg.interval == "1h" else "d"
    for lag in (1, 2, 3, 5, 6, 12, 24):
        name = f"return_{lag}{suffix}"
        f[name] = log_close.diff(lag)
        price_cols.append(name)
    f["lagged_return"] = ret.shift(1)
    f["acceleration"] = ret.diff()
    f["bar_range"] = (f.high - f.low) / f.close
    f["bar_body"] = np.log(f.close / f.open)
    price_cols += ["lagged_return", "acceleration", "bar_range", "bar_body"]
    sent_cols = [
        "sentiment",
        "median_sentiment",
        "sentiment_std",
        "bull_fraction",
        "bear_fraction",
        "neutral_fraction",
        "log_authors",
        "imbalance",
        "has_posts",
        "log_post_count",
        "log_volume_change",
        "sentiment_change",
        "sentiment_lag1",
        "sentiment_lag3",
    ]
    for window in (6, 24, 72):
        vol, trend = f"volatility_{window}{suffix}", f"trend_{window}{suffix}"
        high, low = f"high_{window}", f"low_{window}"
        f[vol] = ret.rolling(window).std(ddof=0)
        f[trend] = f.close / f.close.rolling(window).mean() - 1
        f[high] = f.close / f.high.rolling(window).max() - 1
        f[low] = f.close / f.low.rolling(window).min() - 1
        price_cols.extend([vol, trend, high, low])
        counts = f.post_count.rolling(window).sum()
        level, activity = f"sentiment_{window}{suffix}", f"log_posts_{window}{suffix}"
        f[level] = (f.sentiment.fillna(0) * f.post_count).rolling(window).sum() / counts.replace(0, np.nan)
        f[activity] = np.log1p(counts)
        sent_cols.extend([level, activity])
    f["sentiment_momentum"] = f[f"sentiment_6{suffix}"] - f[f"sentiment_24{suffix}"]
    sent_cols.append("sentiment_momentum")
    f = f.dropna(subset=price_cols)
    if len(f) <= cfg.horizons:
        raise ValueError("Insufficient completed bars after the 72-bar warmup")
    audit.update(
        price_rows=len(p),
        aligned_observations=len(p),
        usable_feature_rows=len(f),
        labeled_observations=len(f) - cfg.horizons,
        first_close_utc=p.index.min().isoformat(),
        last_close_utc=p.index.max().isoformat(),
        missing_price_bars=0,
        hours_with_posts=int(f.has_posts.sum()),
        hours_without_posts=int((f.has_posts == 0).sum()),
        missing_sentiment_intervals_before_warmup_filter=len(p) - len(agg),
        hour_coverage_fraction=float(f.has_posts.mean()),
        engagement_used_as_features=False,
        availability=cfg.availability,
    )
    result = (f, price_cols, price_cols + sent_cols, audit)
    return (*result, x) if with_posts else result


def make_targets(frame, horizons=5):
    return pd.DataFrame(
        {h: np.log(frame.close.shift(-h) / frame.close) for h in range(1, horizons + 1)}, index=frame.index
    )
