"""SQLite storage for runs. One file holds every run; each row carries run_id."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_utc TEXT, mode TEXT, source TEXT,
  start_ts TEXT, end_ts TEXT, models TEXT, config_json TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS bars (
  run_id TEXT, asset TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (run_id, asset, ts));
CREATE TABLE IF NOT EXISTS snapshots (
  run_id TEXT, asset TEXT, ts TEXT, market_hash TEXT, market_json TEXT, numeric_json TEXT,
  PRIMARY KEY (run_id, asset, ts));
CREATE TABLE IF NOT EXISTS decisions (
  run_id TEXT, model TEXT, asset TEXT, ts TEXT, position TEXT, question TEXT,
  p_up REAL, p_act REAL, action TEXT, latency_ms REAL, input_tokens INTEGER,
  cached INTEGER, error TEXT, state_json TEXT, questions_json TEXT,
  PRIMARY KEY (run_id, model, asset, ts));
CREATE TABLE IF NOT EXISTS fills (
  run_id TEXT, model TEXT, asset TEXT, ts TEXT, side TEXT, price REAL, units REAL,
  notional REAL, fee REAL, cash_after REAL, trade_return REAL);
CREATE TABLE IF NOT EXISTS equity (
  run_id TEXT, model TEXT, asset TEXT, ts TEXT, equity REAL, is_long INTEGER,
  PRIMARY KEY (run_id, model, asset, ts));
CREATE TABLE IF NOT EXISTS outcomes (
  run_id TEXT, asset TEXT, ts TEXT, fwd_ret REAL, up_label INTEGER, trade_ret REAL,
  PRIMARY KEY (run_id, asset, ts));
CREATE TABLE IF NOT EXISTS consistency (
  run_id TEXT, model TEXT, asset TEXT, ts TEXT, variant TEXT,
  p_up_orig REAL, p_act_orig REAL, p_up_new REAL, p_act_new REAL, flipped INTEGER, error TEXT);
"""


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def latest_run(conn) -> str | None:
    row = conn.execute("SELECT run_id FROM runs ORDER BY created_utc DESC LIMIT 1").fetchone()
    return row[0] if row else None


def resolve_run(conn, run: str | None) -> str:
    if run in (None, "latest"):
        rid = latest_run(conn)
        if rid is None:
            raise SystemExit("no runs in the database yet; run `python -m tdm replay` first")
        return rid
    return run


def frame(conn, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)
