""""""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    """"""
    date: datetime.date
    weights: Dict[str, float]
    cash_weight: float
    prices: Dict[str, float]            # token → price (USD)
    pre_value: float
    post_value: float
    turnover: float
    transaction_cost: float
    holdings: Dict[str, float]


@dataclass
class PortfolioSnapshot:
    """"""
    date: datetime.date
    total_value: float
    cash: float
    holdings: Dict[str, float]
    weights: Dict[str, float]


class PortfolioManager:
    """"""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 0.0,
        max_weight: float = 1.0,
        max_cash_weight: float = 1.0,
        allow_short: bool = False,
        max_short_weight: float = 0.20,
        name: str = "MRC",
    ):
        self.name = name
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_bps / 10_000.0
        self.max_weight = max_weight
        self.max_cash_weight = max_cash_weight
        self.allow_short = allow_short
        self.max_short_weight = max_short_weight


        self.cash: float = initial_capital
        self.holdings: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {}


        self.rebalance_history: List[RebalanceRecord] = []
        self.portfolio_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def rebalance(
        self,
        weights: Dict[str, float],
        cash_weight: float,
        prices: Dict[str, float],
        date: datetime.date,
    ) -> RebalanceRecord:
        """"""

        pre_value = self._total_value(prices)


        weights, cash_weight = self._clip_weights(weights, cash_weight)


        current_w = self._current_weights(prices, pre_value)


        all_tokens = set(list(weights.keys()) + list(current_w.keys()))
        turnover = sum(abs(weights.get(t, 0.0) - current_w.get(t, 0.0)) for t in all_tokens)


        transaction_cost = pre_value * turnover * self.transaction_cost_rate


        post_value_before_cost = pre_value - transaction_cost
        target_holdings: Dict[str, float] = {}
        for token, w in weights.items():
            price = prices.get(token, 0.0)
            if price > 0 and w != 0:
                target_value = post_value_before_cost * w
                target_holdings[token] = target_value / price
            else:
                target_holdings[token] = 0.0


        self.holdings = {k: v for k, v in target_holdings.items() if v != 0}
        self.cash = post_value_before_cost * cash_weight
        self.current_weights = dict(weights)


        post_value = self._total_value(prices)

        record = RebalanceRecord(
            date=date,
            weights=dict(weights),
            cash_weight=cash_weight,
            prices=dict(prices),
            pre_value=pre_value,
            post_value=post_value,
            turnover=round(turnover, 6),
            transaction_cost=round(transaction_cost, 2),
            holdings=dict(self.holdings),
        )
        self.rebalance_history.append(record)


        self.portfolio_history.append({
            "date": date,
            "total_value": post_value,
            "cash": self.cash,
            "weights": dict(weights),
            "cash_weight": cash_weight,
            "turnover": round(turnover, 6),
            "transaction_cost": round(transaction_cost, 2),
            "event": "rebalance",
        })

        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        all_w = " ".join(f"{tok}:{w:.3f}" for tok, w in sorted_w)
        logger.info(
            f"[Portfolio/{self.name}] Rebalanced on {date}: "
            f"pre=${pre_value:,.0f} → post=${post_value:,.0f} | "
            f"turnover={turnover:.3f} | cost=${transaction_cost:,.0f} | "
            f"cash={cash_weight:.3f} | weights=[{all_w}]"
        )
        return record

    def mark_to_market(self, prices: Dict[str, float], date: datetime.date) -> float:
        """"""
        total = self._total_value(prices)
        self.portfolio_history.append({
            "date": date,
            "total_value": total,
            "cash": self.cash,
            "weights": self._current_weights(prices, total),
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "event": "mark",
        })
        return total

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _total_value(self, prices: Dict[str, float]) -> float:
        """"""
        holdings_value = sum(
            qty * prices.get(sym, 0.0) for sym, qty in self.holdings.items()
        )
        return self.cash + holdings_value

    def _current_weights(self, prices: Dict[str, float], total: float) -> Dict[str, float]:
        """"""
        if total <= 0:
            return {}
        return {
            sym: qty * prices.get(sym, 0.0) / total
            for sym, qty in self.holdings.items()
        }

    def _clip_weights(
        self,
        weights: Dict[str, float],
        cash_weight: float,
    ) -> tuple:
        """"""
        min_w = -self.max_short_weight if self.allow_short else 0.0
        max_w = self.max_weight
        clipped = {k: max(min_w, min(max_w, v)) for k, v in weights.items()}
        cash = max(0.0, min(self.max_cash_weight, float(cash_weight)))

        target_token_sum = 1.0 - cash
        net_token = sum(clipped.values())

        if abs(net_token) > 1e-9:
            scale = target_token_sum / net_token
            clipped = {k: round(v * scale, 8) for k, v in clipped.items()}
        else:
            n = len(clipped) or 1
            clipped = {k: round(target_token_sum / n, 8) for k in clipped}


        for _ in range(20):
            clipped = {k: max(min_w, min(max_w, v)) for k, v in clipped.items()}
            deficit = target_token_sum - sum(clipped.values())
            if abs(deficit) < 1e-9:
                break
            room = {
                k: (max_w - v) if deficit > 0 else (v - min_w)
                for k, v in clipped.items()
                if (deficit > 0 and v < max_w - 1e-9) or (deficit < 0 and v > min_w + 1e-9)
            }
            total_room = sum(room.values())
            if total_room < 1e-9:
                break
            for k, r in room.items():
                clipped[k] = round(clipped[k] + deficit * (r / total_room), 8)

        return clipped, cash

    def get_total_value(self, prices: Dict[str, float]) -> float:
        return self._total_value(prices)

    def compute_daily_returns(self) -> List[float]:
        """"""
        values = [r["total_value"] for r in self.portfolio_history]
        if len(values) < 2:
            return []
        return [
            (values[i] / values[i - 1] - 1.0)
            for i in range(1, len(values))
        ]

    def get_rebalance_dates(self) -> List[datetime.date]:
        return [r.date for r in self.rebalance_history]

    def get_summary(self) -> str:
        if not self.portfolio_history:
            return f"Portfolio initialized with ${self.initial_capital:,.0f}. No trades yet."
        latest = self.portfolio_history[-1]
        pnl = latest["total_value"] - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100
        return (
            f"Total Value: ${latest['total_value']:,.2f} | "
            f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | "
            f"Cash: ${latest['cash']:,.2f} | "
            f"Rebalances: {len(self.rebalance_history)}"
        )
