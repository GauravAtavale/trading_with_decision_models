"""Deciders: anything that answers the two yes/no questions for a snapshot.

* SystemOneClient  hosted decision APIs that use the TypeSafe request shape
                   (Jev, Microsoft-Decision-1). Responses are cached on disk,
                   so re-running a replay costs nothing.
* MockDecider      offline stand-in with the same interface (tests, demos).
* Baselines        buy-and-hold, random, momentum. No API calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .snapshot import ACT_KEY, UP_KEY, canonical


class DeciderError(RuntimeError):
    pass


@dataclass
class Decision:
    p_up: float | None
    p_act: float | None
    latency_ms: float | None = None
    input_tokens: int = 0
    cached: bool = False
    error: str | None = None
    raw: dict | None = None


@dataclass
class Context:
    asset: str
    state: dict             # full state (market + position) as a dict
    encoded_state: object   # what is sent as `state`
    questions: dict
    numeric: dict           # numeric features for baselines/mock (never sent to APIs)
    is_long: bool
    market_hash: str


# ------------------------------------------------------------------ response cache
class ResponseCache:
    """Disk cache of raw API responses keyed by (endpoint, model, payload)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, "
                          "endpoint TEXT, model TEXT, response TEXT, created REAL)")
        self.lock = threading.Lock()

    @staticmethod
    def key(endpoint: str, model: str, payload: dict) -> str:
        return hashlib.sha256(f"{endpoint}|{model}|{canonical(payload)}".encode()).hexdigest()

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, endpoint: str, model: str, response: dict):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                              (key, endpoint, model, json.dumps(response), time.time()))
            self.conn.commit()


# ------------------------------------------------------------------ parsing
PROB_KEYS = ("noul", "probability", "p", "value", "yes", "true")


def parse_noul(answer) -> float:
    """P(yes) from a noul answer. Jev returns {"noul": p}; other shapes are accepted too."""
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        p = float(answer)
    elif isinstance(answer, dict):
        p = None
        for k in PROB_KEYS:
            v = answer.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                p = float(v)
                break
        if p is None:
            probs = answer.get("probabilities") or answer.get("scores")
            if isinstance(probs, dict):
                for k in ("true", "yes", "True"):
                    if isinstance(probs.get(k), (int, float)):
                        p = float(probs[k])
                        break
        if p is None:
            raise DeciderError(f"no probability in answer {str(answer)[:200]}")
    else:
        raise DeciderError(f"unexpected answer {str(answer)[:200]}")
    if not (0.0 <= p <= 1.0) or math.isnan(p):
        raise DeciderError(f"probability out of range: {p}")
    return p


def parse_response(data: dict) -> tuple[float, float, int]:
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise DeciderError(f"response has no answers object: {str(data)[:200]}")
    usage = data.get("usage") or {}
    tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    return parse_noul(answers.get(UP_KEY)), parse_noul(answers.get(ACT_KEY)), tokens


# ------------------------------------------------------------------ hosted APIs
def resolve_provider(name: str, spec: dict) -> dict:
    """Pick the first configured provider whose key, URL and model are all available."""
    tried = []
    for prov in spec.get("providers", []):
        p = spec.get(prov, {})
        key = os.environ.get(p.get("key_env", ""), "")
        url = p.get("url") or (os.environ.get(p.get("url_env", ""), "").rstrip("/") + p.get("url_suffix", "")
                               if os.environ.get(p.get("url_env", "")) else "")
        model = p.get("model") or os.environ.get(p.get("model_env", ""), "")
        missing = [label for label, val, env in (("key", key, p.get("key_env")),
                                                  ("url", url, p.get("url_env")),
                                                  ("model", model, p.get("model_env"))) if not val]
        if not missing:
            return {"provider": prov, "url": url, "model": model, "key": key,
                    "auth": p.get("auth", "bearer")}
        envs = [p.get(f"{m}_env") for m in missing if p.get(f"{m}_env")]
        tried.append(f"{prov} (set {', '.join(envs)})")
    raise DeciderError(f"no credentials for model '{name}'. Options: " + "; ".join(tried))


class SystemOneClient:
    kind = "api"

    def __init__(self, name: str, spec: dict, cache: ResponseCache, max_calls: int = 5000,
                 timeout: float = 30.0):
        self.name = name
        p = resolve_provider(name, spec)
        self.provider, self.url, self.model, self.key, self.auth = (
            p["provider"], p["url"], p["model"], p["key"], p["auth"])
        self.cache, self.max_calls, self.timeout = cache, max_calls, timeout
        self.calls = 0
        self.use_criteria = True
        self.lock = threading.Lock()

    def describe(self) -> str:
        return f"{self.name}: {self.provider} {self.url} (model {self.model})"

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth == "api-key":
            h["api-key"] = self.key
        else:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def post(self, payload: dict, attempts: int = 4) -> dict:
        body = json.dumps(payload).encode()
        delay = 0.5
        for attempt in range(attempts):
            req = urllib.request.Request(self.url, body, self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if not (e.code == 429 or e.code >= 500) or attempt == attempts - 1:
                    raise DeciderError(f"HTTP {e.code}: {detail}") from None
                wait = float(e.headers.get("retry-after") or delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == attempts - 1:
                    raise DeciderError(f"connection failed: {e}") from None
                wait = delay
            time.sleep(wait + random.random() * 0.2)
            delay = min(delay * 2, 8)

    def _payload(self, ctx: Context, criteria: bool | None = None) -> dict:
        questions = ctx.questions
        if not (self.use_criteria if criteria is None else criteria):
            questions = {k: {kk: vv for kk, vv in q.items() if kk != "criteria"} for k, q in questions.items()}
        return {"model": self.model, "state": ctx.encoded_state, "questions": questions}

    def _from_cache(self, ctx: Context):
        """Look up this request with criteria, then without (a deployment may have rejected them)."""
        variants = [True, False] if self.use_criteria else [False]
        for crit in variants:
            hit = self.cache.get(ResponseCache.key(self.url, self.model, self._payload(ctx, crit)))
            if hit is None:
                continue
            try:
                p_up, p_act, tok = parse_response(hit["response"])
            except DeciderError:
                continue  # bad cached entry: ask again
            if not crit and self.use_criteria:
                with self.lock:
                    self.use_criteria = False
            return Decision(p_up, p_act, hit.get("latency_ms"), tok, True, None, hit["response"])
        return None

    def decide(self, ctx: Context, use_cache: bool = True) -> Decision:
        if use_cache:
            hit = self._from_cache(ctx)
            if hit is not None:
                return hit
        payload = self._payload(ctx)
        key = ResponseCache.key(self.url, self.model, payload)
        with self.lock:
            if self.calls >= self.max_calls:
                return Decision(None, None, error=f"request cap of {self.max_calls} reached")
            self.calls += 1
        t0 = time.perf_counter()
        try:
            try:
                data = self.post(payload)
            except DeciderError as e:
                # Some deployments reject `criteria`; retry without it and keep it off.
                sent_criteria = any("criteria" in q for q in payload["questions"].values())
                if "422" in str(e) and sent_criteria:
                    with self.lock:
                        if self.use_criteria:
                            print(f"[{self.name}] request rejected ({str(e)[:120]}); dropping criteria")
                            self.use_criteria = False
                    payload = self._payload(ctx)
                    key = ResponseCache.key(self.url, self.model, payload)
                    data = self.post(payload)
                else:
                    raise
            latency = (time.perf_counter() - t0) * 1000
            p_up, p_act, tok = parse_response(data)
        except DeciderError as e:
            return Decision(None, None, (time.perf_counter() - t0) * 1000, error=str(e)[:300])
        self.cache.put(key, self.url, self.model, {"response": data, "latency_ms": latency})
        return Decision(p_up, p_act, latency, tok, False, None, data)


# ------------------------------------------------------------------ offline deciders
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class MockDecider:
    """Deterministic fake model: leans on 1h momentum plus hash-seeded noise."""
    kind = "mock"

    def __init__(self, name="mock", noise=0.6):
        self.name, self.noise = name, noise

    def describe(self):
        return f"{self.name}: offline mock (no API)"

    def decide(self, ctx: Context, use_cache: bool = True) -> Decision:
        seed = int(hashlib.sha256(f"{self.name}|{ctx.market_hash}|{ctx.is_long}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        n = ctx.numeric
        z = 60 * n["ret_1h"] + 40 * n["dev_sma20"] + rng.gauss(0, self.noise)
        p_up = min(max(_sigmoid(z - 0.8), 0.01), 0.99)
        p_act = 1 - p_up if ctx.is_long else min(max(_sigmoid(z), 0.01), 0.99)
        return Decision(p_up, p_act, latency_ms=5.0 + rng.random() * 3)


class BuyHold:
    kind = "baseline"
    name = "buy_hold"

    def describe(self):
        return "buy_hold: buys on the first bar and never sells"

    def decide(self, ctx, use_cache=True):
        return Decision(None, 0.0 if ctx.is_long else 1.0)


class RandomDecider:
    kind = "baseline"
    name = "random"

    def __init__(self, seed=13):
        self.seed = seed

    def describe(self):
        return "random: uniform probabilities (same threshold as the models)"

    def decide(self, ctx, use_cache=True):
        rng = random.Random(f"{self.seed}|{ctx.asset}|{ctx.market_hash}")
        return Decision(rng.random(), rng.random())


class Momentum:
    kind = "baseline"
    name = "momentum"

    def describe(self):
        return "momentum: buy when 1h return, 20-bar trend and its slope are all positive; sell below the 20-bar average"

    def decide(self, ctx, use_cache=True):
        n = ctx.numeric
        up = n["ret_1h"] > 0 and n["dev_sma20"] > 0 and n["slope_sma20"] > 0
        p_up = float(np.clip(0.5 + 30 * n["ret_1h"], 0.05, 0.95))
        if ctx.is_long:
            return Decision(p_up, 0.8 if n["dev_sma20"] < 0 else 0.2)
        return Decision(p_up, 0.8 if up else 0.2)


BASELINES = {"buy_hold": BuyHold, "random": RandomDecider, "momentum": Momentum}


def build_deciders(cfg, names: list[str], cache: ResponseCache, baselines=True):
    deciders = []
    for name in names:
        spec = cfg.models.get(name)
        if spec is None:
            raise DeciderError(f"model '{name}' is not in config.toml (have: {list(cfg.models)})")
        kind = spec.get("kind")
        if kind == "systemone":
            deciders.append(SystemOneClient(name, spec, cache, cfg.max_calls_per_model))
        elif kind == "mock":
            deciders.append(MockDecider(name))
        else:
            raise DeciderError(f"unknown kind {kind!r} for model '{name}'")
    if baselines:
        deciders += [cls() for cls in BASELINES.values()]
    return deciders
