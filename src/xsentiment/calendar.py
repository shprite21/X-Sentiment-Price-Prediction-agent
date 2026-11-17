"""UTC opening/closing pairs on the configured trading calendar."""

import exchange_calendars as xcals
import pandas as pd

from .config import Config, utc


def bars(start, end, cfg: Config) -> pd.DataFrame:
    left, right = utc(start), utc(end)
    step = pd.Timedelta(hours=1) if cfg.interval == "1h" else pd.Timedelta(days=1)
    if cfg.exchange == "24/7":
        starts = pd.date_range(left.floor(step), right.ceil(step), freq=step)
        return pd.DataFrame({"open": starts, "close": starts + step})
    cal = xcals.get_calendar(cfg.exchange)
    schedule = cal.schedule.loc[
        str((left - pd.Timedelta(days=2)).date()) : str((right + pd.Timedelta(days=2)).date())
    ]
    pairs = []
    for r in schedule.itertuples():
        segments = [(r.open, r.close)]
        if cfg.interval == "1h" and pd.notna(r.break_start):
            segments = [(r.open, r.break_start), (r.break_end, r.close)]
        for begin, finish in segments:
            if cfg.interval == "1d":
                pairs.append((begin, finish))
            else:
                pairs.extend(
                    (t, min(t + step, finish))
                    for t in pd.date_range(begin, finish, freq=step, inclusive="left")
                )
    return pd.DataFrame(pairs, columns=["open", "close"])


def future_closes(stamp, cfg: Config):
    timeline = bars(stamp, stamp + pd.Timedelta(days=35), cfg)
    return timeline.loc[timeline.close > stamp, "close"].iloc[: cfg.horizons].tolist()


def annual_periods(stamp, cfg: Config) -> float:
    if cfg.exchange == "24/7":
        return 365 * (24 if cfg.interval == "1h" else 1)
    # Use the complete preceding calendar year, known before the research window.
    year = stamp.year - 1
    left, right = utc(f"{year}-01-01"), utc(f"{year + 1}-01-01")
    schedule = bars(left, right, cfg)
    return float(schedule.close.between(left, right, inclusive="left").sum())
