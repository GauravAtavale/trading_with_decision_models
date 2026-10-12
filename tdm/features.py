"""Turn bars into a market snapshot written in words.

Rules:
* Only data up to and including bar t's close is used (rolling windows look back).
* Arithmetic and comparisons are done here, so the models read conclusions
  ("0.4% above the 20-bar average") instead of doing math on raw numbers.
* No absolute prices or calendar dates, so a model cannot "remember" what
  happened on a given day during a historical replay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def compute_features(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Numeric features for every bar; row t uses bars <= t only."""
    per_hour = max(1, 60 // bar_minutes)
    per_day = 24 * per_hour
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    ret1 = c.pct_change()
    f["ret_bar"] = ret1
    f["ret_1h"] = c / c.shift(per_hour) - 1
    f["ret_4h"] = c / c.shift(4 * per_hour) - 1
    f["ret_24h"] = c / c.shift(per_day) - 1
    sma20, sma80 = c.rolling(20).mean(), c.rolling(80).mean()
    f["dev_sma20"] = c / sma20 - 1
    f["dev_sma80"] = c / sma80 - 1
    f["slope_sma20"] = sma20 / sma20.shift(4) - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi14"] = (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())
    vol16 = ret1.rolling(16).std()
    f["vol_ratio"] = vol16 / vol16.rolling(7 * per_day, min_periods=per_day).median()
    f["bar_range"] = (h - l) / c
    f["volume_ratio"] = v / v.rolling(20).mean()
    lo24, hi24 = l.rolling(per_day).min(), h.rolling(per_day).max()
    f["range24_pos"] = (c - lo24) / (hi24 - lo24).replace(0, np.nan)
    f["range24_width"] = (hi24 - lo24) / c
    f["hour_utc"] = df.index.hour
    f["weekday"] = df.index.weekday
    return f


# ------------------------------------------------------------------ words
def move(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "unknown"
    p = x * 100
    if abs(p) < 0.005:
        return "flat (0.00%)"
    return f"{'up' if p > 0 else 'down'} {abs(p):.2f}%"


def vs_level(x: float, what: str) -> str:
    if not np.isfinite(x):
        return "unknown"
    p = x * 100
    if abs(p) < 0.005:
        return f"at the {what}"
    return f"{abs(p):.2f}% {'above' if p > 0 else 'below'} the {what}"


def slope_word(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x > 0.0005:
        return "rising"
    if x < -0.0005:
        return "falling"
    return "flat"


def rsi_word(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    label = ("oversold" if x < 30 else "bearish" if x < 45 else "neutral" if x <= 55
             else "bullish" if x <= 70 else "overbought")
    return f"{x:.0f}, {label}"


def vol_word(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    label = "calm" if x < 0.7 else "normal" if x <= 1.3 else "elevated" if x <= 2 else "very high"
    return f"{label}, {x:.1f}x the 7-day typical level"


def range_word(pos: float, width: float) -> str:
    if not (np.isfinite(pos) and np.isfinite(width)):
        return "unknown"
    where = ("near the 24h low" if pos < 0.15 else "in the lower part of the 24h range" if pos < 0.4
             else "mid-range" if pos <= 0.6 else "in the upper part of the 24h range" if pos <= 0.85
             else "near the 24h high")
    return f"{where} ({pos * 100:.0f}% of a {width * 100:.2f}% wide range)"


def market_state(asset: str, row: pd.Series, others: dict[str, pd.Series], bar_minutes: int) -> dict:
    """The market part of a snapshot: identical for every model."""
    state = {
        "asset": asset,
        "bar": f"{bar_minutes}-minute",
        "returns": {
            "last_bar": move(row["ret_bar"]),
            "last_1h": move(row["ret_1h"]),
            "last_4h": move(row["ret_4h"]),
            "last_24h": move(row["ret_24h"]),
        },
        "trend": {
            "vs_20bar_average": vs_level(row["dev_sma20"], "20-bar average"),
            "vs_80bar_average": vs_level(row["dev_sma80"], "80-bar average"),
            "20bar_average_direction": slope_word(row["slope_sma20"]),
        },
        "momentum": {"rsi_14": rsi_word(row["rsi14"])},
        "volatility": {
            "recent": vol_word(row["vol_ratio"]),
            "last_bar_range": f"{row['bar_range'] * 100:.2f}%" if np.isfinite(row["bar_range"]) else "unknown",
        },
        "volume": {"last_bar": (f"{row['volume_ratio']:.1f}x the 20-bar average"
                                if np.isfinite(row["volume_ratio"]) else "unknown")},
        "price_location": range_word(row["range24_pos"], row["range24_width"]),
        "time": {
            "hour_utc": int(row["hour_utc"]),
            "weekday": WEEKDAYS[int(row["weekday"])],
            "weekend": bool(int(row["weekday"]) >= 5),
        },
    }
    if others:
        state["other_assets"] = {a: {"last_1h": move(r["ret_1h"]), "last_24h": move(r["ret_24h"])}
                                 for a, r in others.items()}
    return state


REQUIRED = ["ret_bar", "ret_1h", "ret_4h", "ret_24h", "dev_sma20", "dev_sma80", "slope_sma20",
            "rsi14", "vol_ratio", "volume_ratio", "range24_pos"]


def first_complete_index(features: dict[str, pd.DataFrame]) -> int:
    """First row where every asset has every feature (end of warmup)."""
    ok = None
    for f in features.values():
        row_ok = f[REQUIRED].notna().all(axis=1).to_numpy()
        ok = row_ok if ok is None else ok & row_ok
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError("not enough history to compute features; increase warmup_days")
    return int(idx[0])
