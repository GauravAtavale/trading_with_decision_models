"""Replay loop.

For each bar t in the window:
  1. fill orders decided at t-1 at bar t's open
  2. build the market snapshot at bar t's close (same for every model)
  3. ask every model, in parallel, about its own position
  4. turn P(yes) into BUY / SELL / HOLD with the shared threshold
  5. mark every sleeve to bar t's close
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import db as dbm
from .broker import Sleeve
from .clients import Context, Decision, ResponseCache, build_deciders
from .data import load_bars
from .features import compute_features, first_complete_index, market_state
from .snapshot import (ACT_KEY, build_state, encode_state, make_questions, market_hash,
                       position_state)


def decide_action(d: Decision, is_long: bool, threshold: float) -> str:
    if d.error or d.p_act is None:
        return "HOLD"
    if d.p_act >= threshold:
        return "SELL" if is_long else "BUY"
    return "HOLD"


def _numeric(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}


def prepare(cfg, end=None, verbose=True):
    bars = load_bars(cfg, end, verbose)
    index = next(iter(bars.values())).index
    feats = {a: compute_features(df, cfg.bar_minutes) for a, df in bars.items()}
    first = first_complete_index(feats)
    n = len(index)
    window = int(round(cfg.days * cfg.bars_per_day))
    start = max(first, n - window)
    if verbose and n - window < first:
        print(f"  note: only {n - start} bars available after warmup (asked for {window})")
    return bars, feats, index, start


def contexts_for_bar(cfg, t, bars, feats, sleeves, decider_names):
    """Market state per asset and one Context per (decider, asset)."""
    markets, rows = {}, {}
    for a in cfg.assets:
        row = feats[a].iloc[t]
        others = {o: feats[o].iloc[t] for o in cfg.assets if o != a}
        markets[a] = market_state(a, row, others, cfg.bar_minutes)
        rows[a] = row
    ctxs = []
    for name in decider_names:
        for a in cfg.assets:
            sl = sleeves[(name, a)]
            price = float(bars[a]["close"].iloc[t])
            state = build_state(markets[a], position_state(sl, price, cfg.bar_minutes), cfg)
            ctxs.append((name, Context(
                asset=a, state=state, encoded_state=encode_state(state, cfg),
                questions=make_questions(cfg, a, sl.is_long), numeric=_numeric(rows[a]),
                is_long=sl.is_long, market_hash=market_hash(markets[a]))))
    return markets, rows, ctxs


def estimate(cfg, n_decisions, sample_ctx):
    payload = {"model": "x", "state": sample_ctx.encoded_state, "questions": sample_ctx.questions}
    tokens = len(json.dumps(payload)) / 3.5  # rough chars-per-token
    return tokens, tokens * n_decisions


def run_replay(cfg, model_names, end=None, dry_run=False, verbose=True, notes="", use_cache=True):
    bars, feats, index, start = prepare(cfg, end, verbose)
    n = len(index)
    cache = ResponseCache(cfg.db_path.parent / "api_cache.sqlite")
    deciders = build_deciders(cfg, model_names, cache)
    names = [d.name for d in deciders]
    seed_each = cfg.seed_usd / len(cfg.assets)
    fee, slip = cfg.fee_bps / 1e4, cfg.slippage_bps / 1e4
    sleeves = {(nm, a): Sleeve(seed_each, fee, slip) for nm in names for a in cfg.assets}
    decision_bars = list(range(start, n - 1))
    api = [d for d in deciders if d.kind == "api"]

    if verbose:
        print(f"Replay {index[start]:%Y-%m-%d %H:%M} -> {index[n - 1]:%Y-%m-%d %H:%M} UTC, "
              f"{len(decision_bars)} bars x {len(cfg.assets)} assets")
        for d in deciders:
            print(f"  - {d.describe()}")
    _, _, ctxs = contexts_for_bar(cfg, decision_bars[0], bars, feats, sleeves, names)
    per_call, total = estimate(cfg, len(decision_bars) * len(cfg.assets), ctxs[0][1])
    if verbose:
        print(f"  ~{per_call:.0f} input tokens per request, ~{total / 1e6:.2f}M tokens per API model "
              f"(${total / 1e6 * 0.042:.2f} at $0.042/M; cached responses are free)")
    if dry_run:
        name, ctx = ctxs[0]
        print("\nSample request (first bar, flat position):")
        print(json.dumps({"model": "<model>", "state": ctx.encoded_state, "questions": ctx.questions},
                         indent=2)[:4000])
        return None

    conn = dbm.connect(cfg.db_path)
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{cfg.source}"
    cfg_dump = {k: (str(v) if not isinstance(v, (int, float, str, list, dict, bool)) else v)
                for k, v in vars(cfg).items() if k not in ("raw", "models")}
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", (
        run_id, datetime.now(timezone.utc).isoformat(), "replay", cfg.source,
        str(index[start]), str(index[n - 1]), json.dumps(names), json.dumps(cfg_dump), notes))
    for a in cfg.assets:
        b = bars[a].iloc[start:]
        conn.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
                         [(run_id, a, str(ts), r.open, r.high, r.low, r.close, r.volume)
                          for ts, r in zip(b.index, b.itertuples())])
        close, open_ = bars[a]["close"].to_numpy(), bars[a]["open"].to_numpy()
        h = cfg.horizon_bars
        out = []
        for t in range(start, n - h):
            fwd = close[t + h] / close[t] - 1
            out.append((run_id, a, str(index[t]), fwd, int(fwd * 100 > cfg.up_threshold_pct),
                        close[t + h] / open_[t + 1] - 1))
        conn.executemany("INSERT INTO outcomes VALUES (?,?,?,?,?,?)", out)
    conn.commit()

    pending: dict = {}
    by_name = {d.name: d for d in deciders}
    t_start = time.time()
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        for t in range(start, n):
            ts = str(index[t])
            # 1. fills at this bar's open
            for (nm, a), action in pending.items():
                f = sleeves[(nm, a)].execute(action, float(bars[a]["open"].iloc[t]))
                if f:
                    conn.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                 (run_id, nm, a, ts, f.side, f.price, f.units, f.notional,
                                  f.fee, f.cash_after, f.trade_return))
            pending = {}
            for sl in sleeves.values():
                if sl.is_long:
                    sl.bars_held += 1
            # 2-4. decisions at this bar's close
            if t < n - 1:
                markets, rows, ctxs = contexts_for_bar(cfg, t, bars, feats, sleeves, names)
                for a in cfg.assets:
                    conn.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?)",
                                 (run_id, a, ts, market_hash(markets[a]), json.dumps(markets[a]),
                                  json.dumps(_numeric(rows[a]))))
                futures = {}
                for nm, ctx in ctxs:
                    d = by_name[nm]
                    if d.kind == "api":
                        futures[(nm, ctx.asset)] = (ctx, pool.submit(d.decide, ctx, use_cache))
                    else:
                        futures[(nm, ctx.asset)] = (ctx, d.decide(ctx))
                for (nm, a), (ctx, fut) in futures.items():
                    dec = fut.result() if hasattr(fut, "result") else fut
                    errors += bool(dec.error)
                    action = decide_action(dec, ctx.is_long, cfg.action_threshold)
                    if action != "HOLD":
                        pending[(nm, a)] = action
                    keep_state = by_name[nm].kind != "baseline"
                    conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        run_id, nm, a, ts, "long" if ctx.is_long else "flat",
                        "exit_long" if ctx.is_long else "enter_long",
                        dec.p_up, dec.p_act, action, dec.latency_ms, dec.input_tokens,
                        int(dec.cached), dec.error,
                        json.dumps(ctx.state) if keep_state else None,
                        json.dumps(ctx.questions) if keep_state else None))
            # 5. mark to close
            for (nm, a), sl in sleeves.items():
                conn.execute("INSERT INTO equity VALUES (?,?,?,?,?,?)",
                             (run_id, nm, a, ts, sl.equity(float(bars[a]["close"].iloc[t])), int(sl.is_long)))
            if verbose and ((t - start) % cfg.bars_per_day == 0 or t == n - 1):
                conn.commit()
                eq = {nm: sum(sleeves[(nm, a)].equity(float(bars[a]["close"].iloc[t])) for a in cfg.assets)
                      for nm in names}
                calls = sum(getattr(d, "calls", 0) for d in api)
                print(f"  {index[t]:%Y-%m-%d %H:%M}  " + "  ".join(f"{k} ${v:,.2f}" for k, v in eq.items())
                      + (f"  | API calls {calls}, errors {errors}" if api else "")
                      + f"  [{time.time() - t_start:.0f}s]")
    conn.commit()
    if verbose:
        print(f"\nRun {run_id} saved to {cfg.db_path}")
    return run_id
