"""Load config.toml and .env files."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(paths=None):
    """Read KEY=VALUE lines from .env files; real environment variables win."""
    for folder in dict.fromkeys(paths or [Path.cwd(), ROOT]):
        path = Path(folder) / ".env"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            os.environ.setdefault(key, value)


@dataclass
class Config:
    raw: dict
    assets: list[str]
    bar_minutes: int
    source: str
    data_dir: Path
    days: float
    warmup_days: float
    seed_usd: float
    fee_bps: float
    slippage_bps: float
    horizon_bars: int
    up_threshold_pct: float
    action_threshold: float
    state_format: str
    max_calls_per_model: int
    workers: int
    db_path: Path
    models: dict = field(default_factory=dict)

    @property
    def bars_per_day(self) -> int:
        return 24 * 60 // self.bar_minutes

    @property
    def side_cost(self) -> float:
        """Cost of one side of a trade as a fraction (fee + slippage)."""
        return (self.fee_bps + self.slippage_bps) / 1e4

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.side_cost


def load_config(path: str | Path | None = None, **overrides) -> Config:
    load_dotenv()
    path = Path(path) if path else ROOT / "config.toml"
    raw = tomllib.loads(path.read_text())
    m, r, c, d, run = (raw.get(k, {}) for k in ("market", "replay", "costs", "decision", "run"))
    cfg = Config(
        raw=raw,
        assets=list(m.get("assets", ["BTC-USD", "ETH-USD"])),
        bar_minutes=int(m.get("bar_minutes", 15)),
        source=m.get("source", "coinbase"),
        data_dir=ROOT / m.get("data_dir", "data"),
        days=float(r.get("days", 7)),
        warmup_days=float(r.get("warmup_days", 8)),
        seed_usd=float(r.get("seed_usd", 100.0)),
        fee_bps=float(c.get("fee_bps", 10)),
        slippage_bps=float(c.get("slippage_bps", 2)),
        horizon_bars=int(d.get("horizon_bars", 4)),
        up_threshold_pct=float(d.get("up_threshold_pct", 0.3)),
        action_threshold=float(d.get("action_threshold", 0.6)),
        state_format=d.get("state_format", "json_string"),
        max_calls_per_model=int(run.get("max_calls_per_model", 5000)),
        workers=int(run.get("workers", 8)),
        db_path=ROOT / run.get("db_path", "runs/arena.sqlite"),
        models=raw.get("models", {}),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg
