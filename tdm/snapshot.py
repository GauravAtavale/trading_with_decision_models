"""Build the per-model request: shared market state + that model's own position,
plus two yes/no questions.

Questions (both `noul`, so there is no option order to bias the answer):
  up_next  pure forecast, identical for every model and every position;
           scored for calibration (Brier, log-loss).
  act      enter_long when flat, exit_long when long.
"""
from __future__ import annotations

import hashlib
import json

UP_KEY, ACT_KEY = "up_next", "act"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def market_hash(market: dict) -> str:
    return hashlib.sha256(canonical(market).encode()).hexdigest()[:16]


def position_state(sleeve, price: float, bar_minutes: int) -> dict:
    """The model's own position in this asset, marked at the current close."""
    if sleeve.units <= 0:
        return {"status": "flat, holding only cash", "cash": f"${sleeve.cash:.2f}"}
    # what selling at this price would return, net of the exit cost
    pnl = sleeve.units * price * (1 - sleeve.side_cost) / sleeve.cost_basis - 1
    held = sleeve.bars_held * bar_minutes
    held_txt = f"{held // 60} h {held % 60} min" if held >= 60 else f"{held} min"
    return {
        "status": "long, fully invested",
        "value": f"${sleeve.units * price:.2f}",
        "unrealized_pnl_after_exit_cost": ("up " if pnl >= 0 else "down ") + f"{abs(pnl) * 100:.2f}%",
        "held_for": held_txt,
    }


def build_state(market: dict, position: dict, cfg) -> dict:
    return {"market": market, "your_position": position,
            "trading_cost": f"{cfg.round_trip_cost * 100:.2f}% for a full round trip (buy + sell)"}


def make_questions(cfg, asset: str, is_long: bool, with_criteria: bool = True) -> dict:
    mins = cfg.horizon_bars * cfg.bar_minutes
    horizon = f"{mins} minutes" if mins < 120 else f"{mins // 60} hours"
    rt = f"{cfg.round_trip_cost * 100:.2f}%"
    up = {
        "type": "noul",
        "instructions": (f"Will the {asset} price be more than {cfg.up_threshold_pct:g}% higher "
                         f"{horizon} from now than it is now?"),
    }
    if is_long:
        act = {
            "type": "noul",
            "instructions": (f"You hold {asset}. Should you sell it now? Answer yes if holding it "
                             f"for the next {horizon} is more likely to lose money than to make money."),
        }
        crit = ("Selling now is the better choice", "Keeping the position is the better choice")
    else:
        act = {
            "type": "noul",
            "instructions": (f"You hold only cash. Should you buy {asset} now? Answer yes only if the "
                             f"expected gain over the next {horizon} is larger than the {rt} "
                             f"round-trip trading cost."),
        }
        crit = ("Buying now is the better choice", "Staying in cash is the better choice")
    if with_criteria:
        up["criteria"] = {"true": f"Rises by more than {cfg.up_threshold_pct:g}%",
                          "false": f"Does not rise by more than {cfg.up_threshold_pct:g}%"}
        act["criteria"] = {"true": crit[0], "false": crit[1]}
    return {UP_KEY: up, ACT_KEY: act}


def encode_state(state: dict, cfg):
    """What goes into the request's `state` field."""
    if cfg.state_format == "object":
        return state
    return json.dumps(state, indent=1)
