"""Text/markdown report for a run, plus the consistency check."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import db as dbm
from . import metrics as M
from .clients import Context, ResponseCache, build_deciders
from .snapshot import encode_state


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    int_cols = {c for c in df.columns if pd.api.types.is_integer_dtype(df[c])}
    for idx, row in zip(df.index, df.itertuples(index=False)):
        cells = [str(idx)]
        for col, v in zip(df.columns, row):
            if col in int_cols:
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("–" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(conn, run_id: str, compare: tuple[str, str] | None = None) -> str:
    d = M.load_run(conn, run_id)
    run, cfg = d["run"], d["cfg"]
    api_models = [m for m in d["models"] if m not in ("buy_hold", "random", "momentum")]
    parts = [
        f"# Run {run_id}",
        f"Window: {run['start_ts']} → {run['end_ts']} UTC · source: {run['source']} · "
        f"{cfg.get('bar_minutes')}-min bars · assets: {', '.join(cfg.get('assets', []))}",
        f"Costs: {cfg.get('fee_bps')} bps fee + {cfg.get('slippage_bps')} bps slippage per side · "
        f"action threshold {cfg.get('action_threshold')} · horizon {cfg.get('horizon_bars')} bars · "
        f"seed ${cfg.get('seed_usd')} per model",
        "", "## Trading results", md_table(M.trading_table(d)),
        "", "## Decision quality",
        "Brier and log-loss score the `up_next` forecast (lower is better; compare Brier with "
        "`brier_base_rate`, which always predicts the window's up-rate). Regret is the return given up "
        "versus the better of the two actions available at each bar.",
        md_table(M.quality_table(d), "{:,.3f}"),
        "", "## Reliability", md_table(M.reliability_table(d)),
        "", "## Agreement on `up_next` (% of bars leaning the same way)", md_table(M.agreement_table(d), "{:.0f}"),
    ]
    pairs = [compare] if compare else []
    if not pairs and len(api_models) >= 2:
        pairs = [(api_models[0], api_models[1])]
    for m in api_models:
        pairs.append((m, "buy_hold"))
    if pairs:
        parts += ["", "## Is the gap real? (paired block bootstrap)"]
        for a, b in pairs:
            if a not in d["models"] or b not in d["models"]:
                continue
            c = M.paired_comparison(d, a, b)
            line = (f"- **{a} vs {b}**: return gap {c['return_diff_pp']:+.2f} pp over {c['bars']} bars; "
                    f"per-bar gap {c['per_bar_diff_bps']:+.2f} bps, 95% CI "
                    f"[{c['ci95_bps'][0]:+.2f}, {c['ci95_bps'][1]:+.2f}]; "
                    f"P({a} not better) ≈ {c['p_a_not_better']:.2f}")
            if "brier_gain" in c and np.isfinite(c["brier_gain"]):
                line += (f". Forecast: Brier gain {c['brier_gain']:+.4f}, 95% CI "
                         f"[{c['brier_gain_ci95'][0]:+.4f}, {c['brier_gain_ci95'][1]:+.4f}]")
            parts.append(line)
        parts.append("A CI that includes 0 means this window can't separate the two.")
    return "\n".join(parts) + "\n"


def write_report(conn, run_id, out_dir: Path, compare=None) -> Path:
    text = build_report(conn, run_id, compare)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(text)
    return path


def _shuffle(obj, rng):
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: _shuffle(v, rng) for k, v in items}
    return obj


def run_consistency(cfg, conn, run_id: str, model: str, n: int = 50, seed: int = 0, verbose=True):
    """Resend a sample of past snapshots with JSON keys shuffled and count decision flips."""
    rows = dbm.frame(conn, "SELECT * FROM decisions WHERE run_id=? AND model=? AND state_json IS NOT NULL "
                     "AND error IS NULL", (run_id, model))
    if rows.empty:
        raise SystemExit(f"no stored states for {model} in run {run_id}")
    rows = rows.sample(min(n, len(rows)), random_state=seed)
    snaps = dbm.frame(conn, "SELECT asset, ts, market_hash, numeric_json FROM snapshots WHERE run_id=?",
                      (run_id,)).set_index(["asset", "ts"])
    cache = ResponseCache(cfg.db_path.parent / "api_cache.sqlite")
    decider = build_deciders(cfg, [model], cache, baselines=False)[0]
    rng = random.Random(seed)
    thr = cfg.action_threshold
    flips = 0
    for r in rows.itertuples():
        state = _shuffle(json.loads(r.state_json), rng)
        questions = json.loads(r.questions_json)
        snap = snaps.loc[(r.asset, r.ts)]
        ctx = Context(asset=r.asset, state=state, encoded_state=encode_state(state, cfg),
                      questions=questions, numeric=json.loads(snap.numeric_json),
                      is_long=r.position == "long", market_hash=snap.market_hash)
        dec = decider.decide(ctx, use_cache=False)
        flipped = None if dec.error else int((dec.p_act >= thr) != (r.p_act >= thr))
        flips += flipped or 0
        conn.execute("INSERT INTO consistency VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (run_id, model, r.asset, r.ts, "shuffled_keys", r.p_up, r.p_act,
                      dec.p_up, dec.p_act, flipped, dec.error))
    conn.commit()
    if verbose:
        print(f"{model}: {flips}/{len(rows)} action flips after shuffling JSON key order")
    return flips, len(rows)
