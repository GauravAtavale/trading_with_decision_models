"""Paper broker: long-only, all-in / all-out, one sleeve per (model, asset).

Fills happen at the NEXT bar's open (the decision is made at bar t's close),
with slippage applied to the price and a fee charged on the notional.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fill:
    side: str
    price: float
    units: float
    notional: float
    fee: float
    cash_after: float
    trade_return: float | None = None  # set on sells: net round-trip return


@dataclass
class Sleeve:
    cash: float
    fee_rate: float
    slip_rate: float
    units: float = 0.0
    cost_basis: float = 0.0
    bars_held: int = 0
    fees_paid: float = 0.0
    fills: list = field(default_factory=list)

    @property
    def side_cost(self) -> float:
        return self.fee_rate + self.slip_rate

    @property
    def is_long(self) -> bool:
        return self.units > 0

    @property
    def entry_price(self) -> float:
        """Break-even-style entry: cash spent per unit."""
        return self.cost_basis / self.units if self.units else 0.0

    def equity(self, price: float) -> float:
        """Mark-to-market at `price` (no exit cost deducted)."""
        return self.cash + self.units * price

    def buy(self, open_price: float) -> Fill | None:
        if self.is_long or self.cash <= 0:
            return None
        price = open_price * (1 + self.slip_rate)
        fee = self.cash * self.fee_rate
        units = (self.cash - fee) / price
        self.cost_basis = self.cash
        self.units, self.cash = units, 0.0
        self.fees_paid += fee
        self.bars_held = 0
        f = Fill("BUY", price, units, units * price, fee, self.cash)
        self.fills.append(f)
        return f

    def sell(self, open_price: float) -> Fill | None:
        if not self.is_long:
            return None
        price = open_price * (1 - self.slip_rate)
        notional = self.units * price
        fee = notional * self.fee_rate
        self.cash = notional - fee
        ret = self.cash / self.cost_basis - 1 if self.cost_basis else None
        f = Fill("SELL", price, self.units, notional, fee, self.cash, ret)
        self.units, self.cost_basis, self.bars_held = 0.0, 0.0, 0
        self.fees_paid += fee
        self.fills.append(f)
        return f

    def execute(self, action: str, open_price: float) -> Fill | None:
        if action == "BUY":
            return self.buy(open_price)
        if action == "SELL":
            return self.sell(open_price)
        return None
