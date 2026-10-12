"""Command line.

  python -m tdm fetch                         download/refresh the candle cache
  python -m tdm replay --models jev,microsoft replay the last N days (default from config)
  python -m tdm replay --dry-run              show the request and a cost estimate, no calls
  python -m tdm probe --model jev             one real request; prints the raw response
  python -m tdm report [--run latest]         write and print the run report
  python -m tdm consistency --model jev       re-ask sampled bars with shuffled keys
  python -m tdm runs                          list runs in the database
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from . import db as dbm
from .config import ROOT, load_config


def _end(arg):
    if not arg:
        return None
    dt = datetime.fromisoformat(arg)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tdm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", choices=["coinbase", "csv", "synthetic"])
    common.add_argument("--days", type=float)
    common.add_argument("--end", help="window end, ISO time in UTC (default: last closed bar)")
    common.add_argument("--db", help="SQLite path (default from config)")

    sub.add_parser("fetch", parents=[common], help="download candles into data/")
    r = sub.add_parser("replay", parents=[common], help="historical replay")
    r.add_argument("--models", default="jev,microsoft", help="comma-separated names from config.toml")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-cache", action="store_true", help="ignore cached API responses")
    r.add_argument("--notes", default="")
    pr = sub.add_parser("probe", parents=[common], help="send one request and print the raw response")
    pr.add_argument("--model", required=True)
    rep = sub.add_parser("report", parents=[common])
    rep.add_argument("--run", default="latest")
    rep.add_argument("--compare", help="two models, e.g. jev,microsoft")
    c = sub.add_parser("consistency", parents=[common])
    c.add_argument("--run", default="latest")
    c.add_argument("--model", required=True)
    c.add_argument("--n", type=int, default=50)
    sub.add_parser("runs", parents=[common])

    a = p.parse_args(argv)
    from pathlib import Path
    cfg = load_config(a.config, source=a.source, days=a.days,
                      db_path=Path(a.db) if getattr(a, "db", None) else None)

    if a.cmd == "fetch":
        from .data import load_bars
        frames = load_bars(cfg, _end(a.end))
        for asset, f in frames.items():
            print(f"{asset}: {len(f)} bars, {f.index.min()} -> {f.index.max()}")

    elif a.cmd == "replay":
        from .engine import run_replay
        from .report import write_report
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        run_id = run_replay(cfg, models, _end(a.end), dry_run=a.dry_run, notes=a.notes,
                            use_cache=not a.no_cache)
        if run_id:
            conn = dbm.connect(cfg.db_path)
            path = write_report(conn, run_id, ROOT / "reports")
            print(path.read_text())
            print(f"Report saved to {path}\nDashboard: streamlit run dashboard.py")

    elif a.cmd == "probe":
        from .clients import ResponseCache, build_deciders, parse_response
        from .engine import contexts_for_bar, prepare
        from .broker import Sleeve
        cfg.days = min(cfg.days, 1)
        bars, feats, index, start = prepare(cfg, _end(a.end))
        cache = ResponseCache(cfg.db_path.parent / "api_cache.sqlite")
        decider = build_deciders(cfg, [a.model], cache, baselines=False)[0]
        print(decider.describe())
        sleeves = {(a.model, x): Sleeve(50, 0, 0) for x in cfg.assets}
        _, _, ctxs = contexts_for_bar(cfg, len(index) - 1, bars, feats, sleeves, [a.model])
        ctx = ctxs[0][1]
        payload = {"model": getattr(decider, "model", a.model), "state": ctx.encoded_state,
                   "questions": ctx.questions}
        print("\nRequest:\n" + json.dumps(payload, indent=2))
        if hasattr(decider, "post"):
            raw = decider.post(payload)
            print("\nRaw response:\n" + json.dumps(raw, indent=2))
            p_up, p_act, tok = parse_response(raw)
            print(f"\nParsed: P(up_next)={p_up:.3f}  P(act)={p_act:.3f}  input_tokens={tok}")
        else:
            dec = decider.decide(ctx)
            print(f"\nParsed: P(up_next)={dec.p_up:.3f}  P(act)={dec.p_act:.3f}")

    elif a.cmd == "report":
        from .report import write_report
        conn = dbm.connect(cfg.db_path)
        run_id = dbm.resolve_run(conn, a.run)
        compare = tuple(a.compare.split(",")) if a.compare else None
        path = write_report(conn, run_id, ROOT / "reports", compare)
        print(path.read_text())
        print(f"Saved to {path}")

    elif a.cmd == "consistency":
        from .report import run_consistency
        conn = dbm.connect(cfg.db_path)
        run_consistency(cfg, conn, dbm.resolve_run(conn, a.run), a.model, a.n)

    elif a.cmd == "runs":
        conn = dbm.connect(cfg.db_path)
        df = dbm.frame(conn, "SELECT run_id, source, start_ts, end_ts, models, notes FROM runs "
                       "ORDER BY created_utc DESC")
        print(df.to_string(index=False) if len(df) else "no runs yet")


if __name__ == "__main__":
    sys.exit(main())
