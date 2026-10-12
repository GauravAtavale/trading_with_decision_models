"""Scoring a run: trading results, decision quality, reliability, and whether
the gap between two models is bigger than luck.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import db as dbm

PRICE_PER_M = {"jev": 0.042, "microsoft": 0.042}  # USD per 1M input tokens (output is free)


def load_run(conn, run_id):
    q = lambda sql: dbm.frame(conn, sql, (run_id,))
    run = q("SELECT * FROM runs WHERE run_id=?")
    if run.empty:
        raise SystemExit(f"run {run_id} not found")
    data = {
        "run": run.iloc[0].to_dict(),
        "equity": q("SELECT * FROM equity WHERE run_id=?"),
        "decisions": q("SELECT run_id, model, asset, ts, position, question, p_up, p_act, action, "
                       "latency_ms, input_tokens, cached, error FROM decisions WHERE run_id=?"),
        "fills": q("SELECT * FROM fills WHERE run_id=?"),
        "outcomes": q("SELECT * FROM outcomes WHERE run_id=?"),
        "bars": q("SELECT * FROM bars WHERE run_id=?"),
        "consistency": q("SELECT * FROM consistency WHERE run_id=?"),
    }
    cfg = json.loads(data["run"]["config_json"])
    data["cfg"] = cfg
    data["models"] = json.loads(data["run"]["models"])
    return data


def bars_per_year(cfg) -> float:
    return 365 * 24 * 60 / int(cfg.get("bar_minutes", 15))


def total_equity(equity: pd.DataFrame) -> pd.DataFrame:
    """Rows = timestamps, columns = models, values = equity summed across assets."""
    return equity.pivot_table(index="ts", columns="model", values="equity", aggfunc="sum").sort_index()


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    return float((series / peak - 1).min())


# ------------------------------------------------------------------ trading
def trading_table(d) -> pd.DataFrame:
    cfg, eq, fills = d["cfg"], total_equity(d["equity"]), d["fills"]
    seed = float(cfg.get("seed_usd", 100.0))
    ann = np.sqrt(bars_per_year(cfg))
    bh = eq["buy_hold"] if "buy_hold" in eq else None
    rows = []
    for m in d["models"]:
        s = eq[m]
        r = s.pct_change().dropna()
        sells = fills[(fills.model == m) & (fills.side == "SELL")]
        long_share = d["equity"][d["equity"].model == m].is_long.mean()
        rows.append({
            "model": m,
            "final_$": s.iloc[-1],
            "return_%": (s.iloc[-1] / seed - 1) * 100,
            "vs_buy_hold_pp": ((s.iloc[-1] - bh.iloc[-1]) / seed * 100) if bh is not None else np.nan,
            "sharpe_ann": (r.mean() / r.std() * ann) if r.std() > 0 else np.nan,
            "max_drawdown_%": max_drawdown(s) * 100,
            "trades": int((fills.model == m).sum()),
            "round_trips": len(sells),
            "win_rate_%": (sells.trade_return > 0).mean() * 100 if len(sells) else np.nan,
            "avg_trade_%": sells.trade_return.mean() * 100 if len(sells) else np.nan,
            "fees_$": fills[fills.model == m].fee.sum(),
            "time_in_market_%": long_share * 100,
        })
    return pd.DataFrame(rows).set_index("model")


# ------------------------------------------------------------------ decision quality
def _joined(d) -> pd.DataFrame:
    return d["decisions"].merge(d["outcomes"][["asset", "ts", "fwd_ret", "up_label", "trade_ret"]],
                                on=["asset", "ts"], how="left")


def regret_frame(d) -> pd.DataFrame:
    """Per decision: value of each action over the horizon vs the one taken."""
    cfg = d["cfg"]
    side = (float(cfg["fee_bps"]) + float(cfg["slippage_bps"])) / 1e4
    j = _joined(d).dropna(subset=["trade_ret"]).copy()
    flat = j.position == "flat"
    j["v_hold"] = np.where(flat, 0.0, j.trade_ret)
    j["v_trade"] = np.where(flat, j.trade_ret - 2 * side, -side)  # BUY when flat, SELL when long
    j["v_chosen"] = np.where(j.action == "HOLD", j.v_hold, j.v_trade)
    j["v_best"] = j[["v_hold", "v_trade"]].max(axis=1)
    j["regret"] = j.v_best - j.v_chosen
    j["correct"] = j.v_chosen >= j.v_best - 1e-12
    return j


def calibration_bins(p: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p.values, "y": y.values})
    g = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
    g["range"] = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in g.index]
    return g.reset_index(drop=True)


def quality_table(d) -> pd.DataFrame:
    j = _joined(d)
    rg = regret_frame(d)
    rows = []
    for m in d["models"]:
        x = j[(j.model == m)].dropna(subset=["p_up", "up_label"])
        r = rg[rg.model == m]
        row = {"model": m, "forecasts": int(len(x))}
        if len(x):
            p, y = x.p_up.clip(1e-6, 1 - 1e-6), x.up_label.astype(float)
            base = y.mean()
            cal = calibration_bins(x.p_up, y)
            row.update({
                "brier": ((p - y) ** 2).mean(),
                "brier_base_rate": ((base - y) ** 2).mean(),
                "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
                "ece": (cal.n * (cal.mean_p - cal.observed).abs()).sum() / cal.n.sum(),
                "mean_p_up": x.p_up.mean(),
                "up_rate": base,
                "hit_rate_confident_%": _confident_hits(x) * 100,
            })
        if len(r):
            row.update({"action_correct_%": r.correct.mean() * 100,
                        "avg_regret_bps": r.regret.mean() * 1e4})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _confident_hits(x, lo=0.35, hi=0.65) -> float:
    conf = x[(x.p_up >= hi) | (x.p_up <= lo)]
    if conf.empty:
        return np.nan
    return ((conf.p_up >= hi) == (conf.up_label == 1)).mean()


# ------------------------------------------------------------------ reliability
def reliability_table(d) -> pd.DataFrame:
    dec = d["decisions"]
    rows = []
    for m in d["models"]:
        x = dec[dec.model == m]
        live = x[(x.cached == 0) & x.latency_ms.notna()]
        tokens = x.input_tokens.fillna(0).sum()
        row = {"model": m, "requests": int(len(x)), "errors": int(x.error.notna().sum()),
               "error_rate_%": x.error.notna().mean() * 100,
               "cached_%": x.cached.mean() * 100,
               "latency_p50_ms": live.latency_ms.median() if len(live) else np.nan,
               "latency_p95_ms": live.latency_ms.quantile(0.95) if len(live) else np.nan,
               "input_tokens": int(tokens),
               "est_cost_$": tokens / 1e6 * PRICE_PER_M.get(m, 0.0)}
        cons = d["consistency"]
        if not cons.empty and (cons.model == m).any():
            c = cons[(cons.model == m) & cons.error.isna()]
            row["flip_rate_%"] = c.flipped.mean() * 100 if len(c) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def agreement_table(d) -> pd.DataFrame:
    """Same question, same market: how often do two models lean the same way on up_next?"""
    dec = d["decisions"].dropna(subset=["p_up"])
    wide = dec.pivot_table(index=["asset", "ts"], columns="model", values="p_up")
    models = list(wide.columns)
    out = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                out.loc[a, b] = 100.0
                continue
            both = wide[[a, b]].dropna()
            out.loc[a, b] = float(((both[a] >= 0.5) == (both[b] >= 0.5)).mean() * 100) if len(both) else np.nan
    return out


# ------------------------------------------------------------------ significance
def block_bootstrap_mean(x: np.ndarray, block: int = 16, n_boot: int = 4000, seed: int = 0):
    """Circular block bootstrap of the mean (keeps short-range autocorrelation)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return np.nan, (np.nan, np.nan), np.nan
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_le_zero = (means <= 0).mean()
    return x.mean(), (lo, hi), p_le_zero


def paired_comparison(d, a: str, b: str) -> dict:
    """Is model a better than model b on this run? Uses identical bars for both."""
    eq = total_equity(d["equity"])
    ra, rb = eq[a].pct_change().dropna(), eq[b].pct_change().dropna()
    diff = (ra - rb).to_numpy()
    n_bars = len(diff)
    mean, (lo, hi), p = block_bootstrap_mean(diff)
    out = {"a": a, "b": b, "bars": n_bars,
           "return_diff_pp": (eq[a].iloc[-1] - eq[b].iloc[-1]) / eq[a].iloc[0] * 100,
           "per_bar_diff_bps": mean * 1e4, "ci95_bps": (lo * 1e4, hi * 1e4),
           "p_a_not_better": p}
    j = _joined(d).dropna(subset=["p_up", "up_label"])
    w = j.pivot_table(index=["asset", "ts"], columns="model", values="p_up")
    y = j.drop_duplicates(["asset", "ts"]).set_index(["asset", "ts"]).up_label
    if a in w and b in w:
        both = w[[a, b]].dropna().join(y)
        bd = ((both[b] - both.up_label) ** 2 - (both[a] - both.up_label) ** 2).to_numpy()
        m2, (l2, h2), p2 = block_bootstrap_mean(bd)
        out.update({"brier_gain": m2, "brier_gain_ci95": (l2, h2), "p_a_not_better_brier": p2})
    return out
