from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class Config:
    asset: str = "BTC-USD"
    company: str = "Bitcoin"
    ticker: str = "BTC"
    keywords: tuple = ("Bitcoin", "BTC")
    exchange: str = "24/7"
    timezone: str = "UTC"
    interval: str = "1h"
    days: int = 180
    start: str | None = None
    end: str | None = None
    horizons: int = 5
    min_train: int | None = None
    test_size: int = 336
    holdout: int | None = None
    ridge_alpha: float = 100.0
    cost_bps: float = 10.0
    slippage_bps: float = 2.0  # Included in cost_bps.
    seed: int = 42
    sentiment_engine: str = "vader"
    finbert_path: str = "data/models/finbert"
    availability: str = "publication"
    platform: str = "X"
    embargo: int = 0

    def __post_init__(self):
        ZoneInfo(self.timezone)
        object.__setattr__(self, "keywords", tuple(self.keywords))
        if self.interval not in {"1h", "1d"} or self.horizons != 5:
            raise ValueError("Choose 1h or 1d and five forecast horizons")
        if self.availability not in {"publication", "first_seen"} or self.sentiment_engine not in {
            "vader",
            "finbert",
        }:
            raise ValueError("Unknown availability policy or sentiment engine")
        if self.platform not in {"X", "StockTwits"}:
            raise ValueError("Unknown source platform")
        if min(self.days, self.test_size, self.min_train or 1, self.holdout or 1) <= 0 or self.embargo < 0:
            raise ValueError("Invalid validation windows")
        if (
            not self.asset
            or not self.ticker
            or not self.company
            or self.ridge_alpha <= 0
            or not 0 <= self.slippage_bps <= self.cost_bps
        ):
            raise ValueError("Invalid asset, alpha or cost assumptions")

    def to_dict(self):
        return asdict(self)


def utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
