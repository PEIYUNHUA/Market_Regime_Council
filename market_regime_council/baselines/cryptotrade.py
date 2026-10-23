"""
CryptoTrade Baseline (EMNLP 2024)

Reference:
  "CryptoTrade: A Reflective LLM-based Agent to Guide Zero-shot
   Cryptocurrency Trading"
  Han et al., EMNLP 2024
  ACL Anthology: https://aclanthology.org/2024.emnlp-main.63/
  GitHub: Xtra-Computing/CryptoTrade  |  arXiv: 2407.09546

Architecture reproduced here (adapted for 13-token portfolio allocation):
  Single LLM agent with an explicit reflection mechanism.

  Each period:
    1. Run three analyst agents to collect market/on-chain/macro summaries
    2. Prepend the stored reflection from the previous period (which includes
       the forward-looking expectation AND the realized return once filled in)
    3. Single LLM call → outputs portfolio weights + a new forward-looking
       reflection on what it expects this period and why
    4. `record_period_result` appends the realized return to the reflection
       so the next period sees the complete outcome

  Reflection format stored between periods:
    "[Previous] I allocated {top3} (cash={x}) because {reasoning}.
     Expected: {expectation}. → Realized return: {actual:.2%}"

Key difference from MRC: single-agent with a lightweight reflection memory
instead of multi-agent Shapley-weighted synthesis. The reflection acts as
a human-readable planning/error-correction loop across periods.

LLM calls per period: 3 (analyst formatters) + 1 (decision + reflection) = 4.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

from ..agents.market_observer import MarketObserver
from ..agents.onchain_observer import OnchainObserver
from ..agents.environment_observer import EnvironmentObserver
from ..agents.investment_analyst import InvestmentAnalyst
from ..council.regime_council import DateRangeBundle
from ..llm.base import BaseLLM
from ..config import ALL_TOKENS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

CRYPTOTRADE_SYSTEM = """\
You are CryptoTrade, a reflective LLM-based crypto portfolio agent.
You receive current market signals AND a reflection from the previous period
that describes what you expected, what happened, and what you learned.
Use the reflection to improve this period's allocation decision.
Respond ONLY in valid JSON. No analysis text, no markdown, no prose."""

CRYPTOTRADE_USER = """\
=== PREVIOUS PERIOD REFLECTION ===
{reflection}

=== CURRENT PERIOD SIGNALS ===

[Technical & Price Momentum (OHLCV)]:
{tech_report}

[On-Chain Metrics & Prediction Market Sentiment]:
{sentiment_report}

[Macroeconomic Indicators & Fear/Greed Index]:
{macro_report}

Based on the reflection and current signals, allocate a portfolio across these
13 crypto tokens: ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints:
- Each token weight: 0.00 – 0.40
- Cash weight: 0.00 – 0.30
- Sum of all token weights + cash_weight = 1.00

IMPORTANT: Your response must be ONLY a valid JSON object. No prose, no markdown.
Include fields:
  portfolio_weights  (dict, all 13 tokens, values sum to 1 - cash_weight)
  cash_weight        (float, 0.00 – 0.30)
  market_regime      (string: "bull" | "volatile" | "bear")
  reflection         (string, ≤120 chars: what you expect this period and why,
                      referencing any lesson from the previous period reflection)"""

# ---------------------------------------------------------------------------
#  CryptoTradeCouncil
# ---------------------------------------------------------------------------

class CryptoTradeCouncil:
    """
    CryptoTrade (EMNLP 2024) reflective single-agent baseline.

    See module docstring for architecture details.
    """

    def __init__(self, llm: BaseLLM, lookback_days: int = 90):
        self.llm           = llm
        self.lookback_days = lookback_days

        # Analyst agents for data formatting (same as other baselines)
        self.agent1 = MarketObserver(llm=llm)
        self.agent2 = OnchainObserver(llm=llm)
        self.agent3 = EnvironmentObserver(llm=llm)

        self._analyst = InvestmentAnalyst(llm=llm)

        # Reflection state persisted across periods
        self._reflection: str = (
            "No previous period. This is the first allocation decision. "
            "Start with a balanced portfolio and observe the market carefully."
        )
        # Last period's weights (for the reflection text)
        self._last_weights: Dict[str, float] = {}
        self._last_cash: float = 0.0
        self._period_count: int = 0

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        logger.info("[CryptoTrade] run_with_bundle: eval_date=%s", bundle.eval_date)

        # Stage 1: three analyst analyses (3 LLM calls)
        eval1 = await self.agent1.analyze(
            bundle.market_snapshots,
            kline_images=bundle.kline_images,
            lookback_days=self.lookback_days,
        )
        eval2 = await self.agent2.analyze(
            bundle.onchain_metrics,
            btc_macro_df=bundle.btc_macro_df,
        )
        eval3 = await self.agent3.analyze(
            bundle.env_snapshot,
            fred_df=bundle.fred_df,
            fng_df=bundle.fng_df,
            tweets_df=bundle.tweets_df,
            polymarket_df=bundle.polymarket_df,
        )

        tech_compact  = InvestmentAnalyst._compact_stage1(eval1, max_raw=350)
        sent_compact  = InvestmentAnalyst._compact_stage1(eval2, max_raw=350)
        macro_compact = InvestmentAnalyst._compact_stage1(eval3, max_raw=350)

        # Stage 2: CryptoTrade agent with reflection context (1 LLM call)
        response = await self.llm.chat_text(
            system_prompt=CRYPTOTRADE_SYSTEM,
            user_prompt=CRYPTOTRADE_USER.format(
                reflection       = self._reflection,
                tech_report      = tech_compact,
                sentiment_report = sent_compact,
                macro_report     = macro_compact,
            ),
        )

        parsed = self._analyst._parse_json(response)
        norm = self._analyst._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )
        weights    = norm["token_weights"]
        cash_weight = norm["cash_weight"]

        # Build the forward-looking reflection (outcome TBD — filled in record_period_result)
        forward_reflection = parsed.get("reflection", "No reflection provided.")
        top3 = sorted(weights.items(), key=lambda x: -float(x[1]))[:3]
        top3_str = ", ".join(f"{k}={float(v):.3f}" for k, v in top3)
        self._reflection = (
            f"[Period {self._period_count + 1}] Allocated top3=[{top3_str}] "
            f"cash={cash_weight:.3f}. Expectation: {forward_reflection}"
            # realized return will be appended by record_period_result
        )

        self._last_weights = dict(weights)
        self._last_cash    = cash_weight
        self._period_count += 1

        logger.info(
            "[CryptoTrade] Decision done | cash=%.2f period=%d",
            cash_weight, self._period_count,
        )

        return {
            "portfolio_weights": weights,
            "cash_weight":       cash_weight,
            "market_regime":     parsed.get("market_regime", "cryptotrade_decision"),
        }

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        """
        Append realized return to the stored reflection so the next period
        can see the full forward-vs-actual outcome (core of the reflection loop).
        """
        self._reflection += f" → Realized: {actual_return:+.2%}"
        logger.debug("[CryptoTrade] Reflection updated: %s", self._reflection[-80:])
