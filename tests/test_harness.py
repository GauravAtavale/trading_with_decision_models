"""Offline tests: no network, no API keys. Run with `python -m pytest -q`."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from tdm import db as dbm
from tdm import metrics as M
from tdm.broker import Sleeve
from tdm.clients import DeciderError, parse_noul, parse_response
from tdm.config import load_config
from tdm.data import synthetic_bars
from tdm.engine import run_replay
from tdm.features import compute_features, first_complete_index, market_state
from tdm.report import build_report, run_consistency
from tdm.snapshot import ACT_KEY, UP_KEY, make_questions

END = datetime(2026, 10, 12, tzinfo=timezone.utc)


# ------------------------------------------------------------------ features
def test_features_use_no_future_data():
    df = synthetic_bars(["BTC-USD"], datetime(2026, 9, 1, tzinfo=timezone.utc), 1500, 15)["BTC-USD"]
    full = compute_features(df, 15)
    for t in (900, 1100, 1499):
        trunc = compute_features(df.iloc[: t + 1], 15)
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[t], check_names=False)


def test_market_state_has_no_prices_or_dates():
    frames = synthetic_bars(["BTC-USD", "ETH-USD"], datetime(2026, 9, 1, tzinfo=timezone.utc), 1500, 15)
    feats = {a: compute_features(f, 15) for a, f in frames.items()}
    t = first_complete_index(feats)
    state = market_state("BTC-USD", feats["BTC-USD"].iloc[t], {"ETH-USD": feats["ETH-USD"].iloc[t]}, 15)
    text = json.dumps(state)
    assert "2026" not in text and "unknown" not in text
    price = frames["BTC-USD"]["close"].iloc[t]
    assert f"{price:.0f}" not in text


# ------------------------------------------------------------------ broker
def test_broker_round_trip_costs():
    s = Sleeve(cash=100.0, fee_rate=0.001, slip_rate=0.0002)
    s.buy(50.0)
    assert s.is_long and s.cash == 0
    assert s.units == pytest.approx(99.9 / (50 * 1.0002))
    f = s.sell(50.0)
    expected = s.cash
    assert f.trade_return == pytest.approx(expected / 100 - 1)
    # flat price: lose about 2 x (fee + slip)
    assert f.trade_return == pytest.approx(-0.0024, abs=2e-5)
    assert s.buy(10) is not None and s.buy(10) is None  # no double buy


# ------------------------------------------------------------------ parsing
def test_parse_noul_shapes():
    assert parse_noul({"type": "noul", "noul": 0.7}) == 0.7
    assert parse_noul({"probability": 0.2}) == 0.2
    assert parse_noul({"probabilities": {"true": 0.4, "false": 0.6}}) == 0.4
    assert parse_noul(0.9) == 0.9
    with pytest.raises(DeciderError):
        parse_noul({"noul": 1.5})
    with pytest.raises(DeciderError):
        parse_noul({"type": "noul"})
    up, act, tok = parse_response({"answers": {UP_KEY: {"noul": 0.3}, ACT_KEY: {"noul": 0.8}},
                                   "usage": {"input_tokens": 512}})
    assert (up, act, tok) == (0.3, 0.8, 512)


def test_questions_are_yes_no_and_position_dependent():
    cfg = load_config()
    flat = make_questions(cfg, "BTC-USD", is_long=False)
    long = make_questions(cfg, "BTC-USD", is_long=True)
    assert all(q["type"] == "noul" for q in {**flat, **long}.values())
    assert "buy" in flat[ACT_KEY]["instructions"] and "sell" in long[ACT_KEY]["instructions"]
    assert flat[UP_KEY] == long[UP_KEY]


# ------------------------------------------------------------------ fake decision API
class FakeAPI(BaseHTTPRequestHandler):
    log: list = []
    fail_first_with_429 = True
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            FakeAPI.log.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                "api_key": self.headers.get("api-key"), "body": body})
            first = FakeAPI.fail_first_with_429
            FakeAPI.fail_first_with_429 = False
        if first:
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        if self.path == "/v1/systemone":
            if self.headers.get("Authorization") != "Bearer ts-test":
                return self._send(401, {"error": "bad key"})
        elif self.path == "/providers/microsoft/v1/systemone":
            if self.headers.get("api-key") != "az-test":
                return self._send(401, {"error": "bad key"})
            if any("criteria" in q for q in body["questions"].values()):
                return self._send(422, {"error": "criteria not supported"})
        else:
            return self._send(404, {"error": "no route"})
        h = int(hashlib.sha256((self.path + json.dumps(body["state"])).encode()).hexdigest()[:8], 16)
        answers = {name: {"type": "noul", "noul": ((h >> (4 * i)) % 1000) / 1000}
                   for i, name in enumerate(body["questions"])}
        self._send(200, {"model": body["model"], "answers": answers,
                         "usage": {"input_tokens": len(json.dumps(body)) // 4}})

    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_api(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeAPI.log = []
    FakeAPI.fail_first_with_429 = True
    toml = (tmp_path / "config.toml")
    base = open(load_config.__globals__["ROOT"] / "config.toml").read()
    base = base.replace("https://api.typesafe.ai/v1/systemone", f"http://127.0.0.1:{port}/v1/systemone")
    toml.write_text(base)
    for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "AZURE_FOUNDRY_ENDPOINT", "AZURE_API_KEY",
              "MS_DECISION_DEPLOYMENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("AZURE_API_KEY", "az-test")
    monkeypatch.setenv("MS_DECISION_DEPLOYMENT", "my-decision-1")
    cfg = load_config(toml, source="synthetic", days=1.0, db_path=tmp_path / "arena.sqlite")
    yield cfg
    server.shutdown()


def test_replay_against_fake_api(fake_api):
    cfg = fake_api
    run_id = run_replay(cfg, ["jev", "microsoft"], END, verbose=False)
    conn = sqlite3.connect(cfg.db_path)
    dec = dbm.frame(conn, "SELECT * FROM decisions WHERE run_id=?", (run_id,))
    api = dec[dec.model.isin(["jev", "microsoft"])]
    assert len(api) == 2 * 2 * (cfg.bars_per_day - 1)  # 2 models x 2 assets x bars
    assert api.error.isna().all(), api.error.dropna().head()
    assert api.p_up.between(0, 1).all() and api.p_act.between(0, 1).all()

    paths = {r["path"] for r in FakeAPI.log}
    assert paths == {"/v1/systemone", "/providers/microsoft/v1/systemone"}
    ms = [r for r in FakeAPI.log if r["path"].startswith("/providers")]
    assert all(r["api_key"] == "az-test" and r["auth"] is None for r in ms)
    assert ms[-1]["body"]["model"] == "my-decision-1"
    assert all("criteria" not in q for q in ms[-1]["body"]["questions"].values())  # fallback kicked in
    jev = [r for r in FakeAPI.log if r["path"] == "/v1/systemone"]
    assert all(r["auth"] == "Bearer ts-test" for r in jev)
    assert any("criteria" in q for q in jev[-1]["body"]["questions"].values())  # jev accepted criteria

    # every model saw the same market for a given asset and bar
    states = dec.dropna(subset=["state_json"])
    for (asset, ts), g in states.groupby(["asset", "ts"]):
        markets = {json.dumps(json.loads(s)["market"], sort_keys=True) for s in g.state_json}
        assert len(markets) == 1

    # second run: everything served from the cache, no new HTTP calls
    n_calls = len(FakeAPI.log)
    run2 = run_replay(cfg, ["jev", "microsoft"], END, verbose=False)
    assert len(FakeAPI.log) == n_calls
    dec2 = dbm.frame(conn, "SELECT * FROM decisions WHERE run_id=? AND model IN ('jev','microsoft')", (run2,))
    assert dec2.cached.all()
    # identical inputs -> identical decisions and equity
    eq1 = M.total_equity(dbm.frame(conn, "SELECT * FROM equity WHERE run_id=?", (run_id,)))
    eq2 = M.total_equity(dbm.frame(conn, "SELECT * FROM equity WHERE run_id=?", (run2,)))
    pd.testing.assert_frame_equal(eq1, eq2)

    report = build_report(conn, run2)
    assert "jev vs microsoft" in report and "Trading results" in report

    flips, n = run_consistency(cfg, conn, run2, "jev", n=10, verbose=False)
    assert n == 10 and 0 <= flips <= 10


def test_api_errors_become_hold(fake_api, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong-key")
    cfg = fake_api
    run_id = run_replay(cfg, ["jev"], END, verbose=False, use_cache=False)
    conn = sqlite3.connect(cfg.db_path)
    dec = dbm.frame(conn, "SELECT * FROM decisions WHERE run_id=? AND model='jev'", (run_id,))
    assert dec.error.notna().all() and (dec.action == "HOLD").all()
    eq = dbm.frame(conn, "SELECT * FROM equity WHERE run_id=? AND model='jev'", (run_id,))
    assert eq.equity.round(6).eq(cfg.seed_usd / len(cfg.assets)).all()


# ------------------------------------------------------------------ metrics
def test_bootstrap_detects_real_gap_and_not_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1e-3, 700)
    _, (lo, hi), _ = M.block_bootstrap_mean(noise)
    assert lo < 0 < hi
    _, (lo, hi), p = M.block_bootstrap_mean(noise + 5e-4)
    assert lo > 0 and p < 0.05


# ------------------------------------------------------------------ coinbase download (faked)
def test_coinbase_pagination_and_cache(tmp_path, monkeypatch):
    from datetime import timedelta
    from urllib.parse import parse_qs, urlparse
    import tdm.data as D

    calls = []

    def fake_get(url, attempts=5):
        q = parse_qs(urlparse(url).query)
        s, e = (datetime.fromisoformat(q[k][0]) for k in ("start", "end"))
        calls.append((s, e))
        rows, t = [], s
        while t < e:  # Coinbase returns newest first: [time, low, high, open, close, volume]
            ts = int(t.timestamp())
            rows.append([ts, 99.0, 101.0, 100.0, 100.5, 7.0])
            t += timedelta(minutes=15)
        return rows[::-1]

    monkeypatch.setattr(D, "_get_json", fake_get)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    start = datetime(2026, 9, 27, tzinfo=timezone.utc)
    end = start + timedelta(days=15)
    df = D.load_coinbase_cached("BTC-USD", start, end, 15, tmp_path, verbose=False)
    assert len(df) == 15 * 96 and df.index.is_monotonic_increasing
    assert len(calls) == -(-15 * 96 // 300)  # 300 candles per request
    assert (df.close == 100.5).all() and str(df.index.tz) == "UTC"
    # extending the window only downloads the missing tail
    calls.clear()
    df2 = D.load_coinbase_cached("BTC-USD", start, end + timedelta(days=1), 15, tmp_path, verbose=False)
    assert len(df2) == 16 * 96 and len(calls) == 1 and calls[0][0] == end
