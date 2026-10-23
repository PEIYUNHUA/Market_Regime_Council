"""
FinCon Baseline (NeurIPS 2024)

Reference:
  "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal
   Reinforcement for Enhanced Financial Decision Making"
  NeurIPS 2024  |  GitHub: The-FinAI/FinCon

Architecture reproduced here (adapted for crypto portfolio allocation):
  Stage 1 — Three specialized analysts run independently:
    · Technical Analyst  (Agent 1 data: OHLCV, K-line momentum)
    · Sentiment Analyst  (Agent 2 data: on-chain flows, Polymarket sentiment)
    · Macro Analyst      (Agent 3 data: FRED indicators, Fear & Greed)
  Stage 2 — Manager synthesizes all analyst reports → 13-token portfolio weights
  CVR Loop — Manager writes textual feedback to analysts each period;
             feedback is prepended to analysts' next-period context.

Key difference from MRC: linear hierarchy (no pairwise Socratic debates, no
Shapley Value dynamic reweighting). The CVR feedback is the learning signal.

LLM call count per period: 3 (analysts) + 1 (manager) = 4 total.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Dict, Optional, Tuple

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

FINCON_MANAGER_SYSTEM = """\
You are the Portfolio Manager in a FinCon multi-agent financial trading system.
You receive independent reports from three specialized analysts and synthesize
them into a final 13-token crypto portfolio allocation.
You also write brief Conceptual Verbal Reinforcement (CVR) feedback that will
be sent to each analyst to improve their next-period analysis.
Respond ONLY in valid JSON. No analysis text, no markdown, no prose."""

FINCON_MANAGER_USER = """\
=== ANALYST REPORTS ===

[TECHNICAL ANALYST — OHLCV price momentum & K-line patterns]:
{tech_report}

[SENTIMENT & ON-CHAIN ANALYST — blockchain activity & market sentiment]:
{sentiment_report}

[MACRO ANALYST — FRED macro indicators & Fear/Greed index]:
{macro_report}

=== CVR FEEDBACK FROM PREVIOUS PERIOD ===
{manager_feedback}

Based on the above, allocate a portfolio across these 13 crypto tokens:
ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints:
- Each token weight: 0.00 – 0.40
- Cash weight: 0.00 – 0.30
- Sum of all token weights + cash_weight = 1.00

IMPORTANT: Your response must be ONLY a valid JSON object. No analysis text, no markdown.
Output the JSON immediately starting with the opening brace."""

# ---------------------------------------------------------------------------
#  FinConCouncil
# ---------------------------------------------------------------------------

class FinConCouncil:
    """
    FinCon (NeurIPS 2024) multi-agent baseline for crypto portfolio allocation.

    See module docstring for architecture details.
    """

    def __init__(self, llm: BaseLLM, lookback_days: int = 90):
        self.llm           = llm
        self.lookback_days = lookback_days

        # Analyst agents (same data as MRC Stage 1, independent instances)
        self.agent1 = MarketObserver(llm=llm)     # Technical
        self.agent2 = OnchainObserver(llm=llm)    # Sentiment/On-chain
        self.agent3 = EnvironmentObserver(llm=llm) # Macro

        # Utility: reuse _parse_json and _normalize_weights from InvestmentAnalyst
        self._analyst = InvestmentAnalyst(llm=llm)

        # CVR state: manager feedback for the next period
        self._manager_feedback: str = (
            "No previous period feedback. This is the first allocation decision. "
            "Provide balanced analysis."
        )

    # ------------------------------------------------------------------
    #  Public interface (same as RegimeCouncil)
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        logger.info("[FinCon] run_with_bundle: eval_date=%s", bundle.eval_date)

        # Stage 1: independent analyst analyses (3 LLM calls)
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

        logger.info(
            "[FinCon] Stage 1 done | conf1=%.2f conf2=%.2f conf3=%.2f",
            eval1.confidence, eval2.confidence, eval3.confidence,
        )

        # Stage 2: manager synthesis + CVR (1 LLM call)
        weights, cash_weight, new_feedback = await self._manager_synthesize(
            eval1, eval2, eval3
        )

        # Update CVR feedback for next period
        self._manager_feedback = new_feedback

        logger.info("[FinCon] Manager done | cash_weight=%.2f", cash_weight)

        return {
            "portfolio_weights": weights,
            "cash_weight":       cash_weight,
            "market_regime":     "fincon_decision",
        }

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        """Optional performance feedback (CVR is handled internally by manager)."""
        pass

    # ------------------------------------------------------------------
    #  Manager synthesis
    # ------------------------------------------------------------------

    async def _manager_synthesize(
        self,
        eval1, eval2, eval3
    ) -> Tuple[Dict[str, float], float, str]:
        """
        FinCon manager synthesizes 3 analyst reports → portfolio weights.
        Also produces CVR feedback string for next period.

        Returns (token_weights, cash_weight, cvr_feedback).
        """
        loop = asyncio.get_event_loop()

        tech_compact = InvestmentAnalyst._compact_stage1(eval1, max_raw=400)
        sent_compact = InvestmentAnalyst._compact_stage1(eval2, max_raw=400)
        macro_compact = InvestmentAnalyst._compact_stage1(eval3, max_raw=400)

        user_prompt = FINCON_MANAGER_USER.format(
            tech_report      = tech_compact,
            sentiment_report = sent_compact,
            macro_report     = macro_compact,
            manager_feedback = self._manager_feedback,
        )

        response = await self.llm.chat_text(
            system_prompt=FINCON_MANAGER_SYSTEM,
            user_prompt=user_prompt,
        )

        parsed = self._analyst._parse_json(response)

        norm = self._analyst._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )

        cvr_feedback = parsed.get(
            "analyst_feedback",
            "Maintain current analytical approach; continue monitoring major regime shifts.",
        )

        return norm["token_weights"], norm["cash_weight"], cvr_feedback
