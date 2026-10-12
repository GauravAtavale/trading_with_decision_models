"""Market data: Coinbase public candles (cached as CSV), CSV import, or synthetic bars.

Every source returns a DataFrame indexed by UTC bar-open time with columns
open, high, low, close, volume.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_MAX = 300  # candles per request
COINBASE_GRANULARITIES = {1, 5, 15, 60, 360, 1440}  # minutes


def _get_json(url: str, attempts: int = 5):
    headers = {"User-Agent": "trading-with-decision-models/0.1", "Accept": "application/json"}
    delay = 0.5
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:200]!r}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts - 1:
                raise RuntimeError(f"could not reach {url}: {e}") from None
        time.sleep(delay)
        delay = min(delay * 2, 8)


def fetch_coinbase(product: str, start: datetime, end: datetime, bar_minutes: int,
                   verbose: bool = True) -> pd.DataFrame:
    """Download candles [start, end) from Coinbase Exchange's public API (no key needed)."""
    if bar_minutes not in COINBASE_GRANULARITIES:
        raise ValueError(f"Coinbase supports bar_minutes in {sorted(COINBASE_GRANULARITIES)}")
    step = timedelta(minutes=bar_minutes * COINBASE_MAX)
    rows, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        query = urllib.parse.urlencode({
            "granularity": bar_minutes * 60,
            "start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        url = COINBASE_URL.format(product=product) + "?" + query
        data = _get_json(url)
        if isinstance(data, dict):  # error payload
            raise RuntimeError(f"Coinbase error for {product}: {data}")
        rows.extend(data)
        if verbose:
            print(f"  {product}: {cursor:%Y-%m-%d %H:%M} -> {chunk_end:%Y-%m-%d %H:%M} ({len(data)} bars)")
        cursor = chunk_end
        time.sleep(0.15)  # stay well under the public rate limit
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()[COLUMNS].astype(float)
    return df[(df.index >= start) & (df.index < end)]


def cache_path(data_dir: Path, asset: str, bar_minutes: int) -> Path:
    return Path(data_dir) / f"{asset}_{bar_minutes}m.csv"


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol).sort_index()
    df.index.name = "time"
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df[COLUMNS].astype(float)


def load_coinbase_cached(asset: str, start: datetime, end: datetime, bar_minutes: int,
                         data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Return bars for [start, end), downloading only what the CSV cache lacks."""
    path = cache_path(data_dir, asset, bar_minutes)
    cached = read_csv(path) if path.is_file() else pd.DataFrame(columns=COLUMNS)
    parts = [cached]
    if cached.empty:
        parts.append(fetch_coinbase(asset, start, end, bar_minutes, verbose))
    else:
        first, last = cached.index.min(), cached.index.max()
        if start < first:
            parts.append(fetch_coinbase(asset, start, first.to_pydatetime(), bar_minutes, verbose))
        nxt = last.to_pydatetime() + timedelta(minutes=bar_minutes)
        if nxt < end:
            parts.append(fetch_coinbase(asset, nxt, end, bar_minutes, verbose))
    df = pd.concat([p for p in parts if not p.empty])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")
    return df[(df.index >= start) & (df.index < end)]


def synthetic_bars(assets: list[str], start: datetime, n_bars: int, bar_minutes: int,
                   seed: int = 7) -> dict[str, pd.DataFrame]:
    """Correlated random walks with drifting trends and volatility clustering.

    Only for offline testing: the series has some momentum so the harness has
    something to find, but it says nothing about real markets.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_bars, freq=f"{bar_minutes}min", tz="UTC")
    common = rng.standard_normal(n_bars)
    drift = np.zeros(n_bars)
    for i in range(1, n_bars):  # slowly changing trend regime
        drift[i] = 0.995 * drift[i - 1] + 0.00004 * rng.standard_normal()
    out = {}
    for k, asset in enumerate(assets):
        base_vol = 0.0022 * (1 + 0.3 * k)
        vol = np.empty(n_bars)
        vol[0] = base_vol
        shocks = 0.75 * common + np.sqrt(1 - 0.75 ** 2) * rng.standard_normal(n_bars)
        rets = np.empty(n_bars)
        for i in range(n_bars):
            if i:
                vol[i] = np.sqrt(0.9 * vol[i - 1] ** 2 + 0.1 * base_vol ** 2 * (1 + shocks[i - 1] ** 2) / 2)
            rets[i] = drift[i] * (1 + 0.2 * k) + vol[i] * shocks[i]
        close = (60000.0 if k == 0 else 2500.0 * (k + 1)) * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0] / np.exp(rets[0])], close[:-1]])
        wick = np.abs(rng.standard_normal((2, n_bars))) * vol * close * 0.6
        high = np.maximum(open_, close) + wick[0]
        low = np.minimum(open_, close) - wick[1]
        volume = np.exp(rng.normal(3, 0.4, n_bars)) * (1 + 40 * np.abs(rets))
        out[asset] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                                   "volume": volume}, index=index)
    return out


def load_bars(cfg, end: datetime | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Bars for warmup + replay window for every asset, aligned on common timestamps."""
    bm = cfg.bar_minutes
    if end is None:
        now = datetime.now(timezone.utc)
        # last fully closed bar
        end = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % bm)
    total_days = cfg.days + cfg.warmup_days
    start = end - timedelta(days=total_days)
    if cfg.source == "synthetic":
        n = int(total_days * cfg.bars_per_day)
        frames = synthetic_bars(cfg.assets, start, n, bm)
    elif cfg.source == "csv":
        frames = {a: read_csv(cache_path(cfg.data_dir, a, bm)) for a in cfg.assets}
        frames = {a: f[(f.index >= start) & (f.index < end)] for a, f in frames.items()}
    elif cfg.source == "coinbase":
        if verbose:
            print(f"Loading {bm}-minute bars {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC")
        frames = {a: load_coinbase_cached(a, start, end, bm, cfg.data_dir, verbose) for a in cfg.assets}
    else:
        raise ValueError(f"unknown source {cfg.source!r}")
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    frames = {a: f.loc[common] for a, f in frames.items()}
    expected = int(total_days * cfg.bars_per_day)
    if verbose and len(common) < 0.97 * expected:
        print(f"  note: {len(common)} aligned bars, expected about {expected} (gaps in the feed)")
    return frames
