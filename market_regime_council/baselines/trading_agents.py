"""
TradingAgents Baseline (2025)

Reference:
  "TradingAgents: Multi-Agent LLM Financial Trading Framework"
  Yijia Liu et al., 2025  |  GitHub: TauricResearch/TradingAgents

Architecture reproduced here (adapted for crypto portfolio, simplified to 5 roles):
  Stage 1 — Three specialist analysts analyse data independently:
    · Technical Analyst  (OHLCV, price momentum)
    · Sentiment Analyst  (on-chain, Polymarket sentiment)
    · Macro Analyst      (FRED, Fear/Greed)
  Stage 2 — Adversarial Debate:
    · Bull Researcher  reads all analyst reports → argues for aggressive allocation
    · Bear Researcher  reads all analyst reports → argues for defensive allocation
  Stage 3 — Final Decision:
    · Risk Manager  reads both researcher proposals → balanced final allocation

Key difference from MRC: adversarial bull/bear debate rather than Shapley-weighted
Socratic synthesis. No dynamic Shapley reweighting across periods.

LLM call count per period: 3 (analysts) + 1 (bull researcher) + 1 (bear researcher)
                         + 1 (risk manager) = 6 total.
"""
from __future__ import annotations

import asyncio
import datetime
import json
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

BULL_RESEARCHER_SYSTEM = """\
You are the Bull Researcher in a TradingAgents multi-agent trading system.
Your role is to synthesize analyst reports and construct the strongest BULLISH
case for crypto market outperformance, proposing an aggressive portfolio.
Respond ONLY in valid JSON. No prose, no markdown."""

BULL_RESEARCHER_USER = """\
=== ANALYST REPORTS ===

[TECHNICAL ANALYST]:
{tech_report}

[SENTIMENT & ON-CHAIN ANALYST]:
{sentiment_report}

[MACRO ANALYST]:
{macro_report}

As the Bull Researcher, identify the strongest bullish signals and propose an
aggressive portfolio allocation across 13 crypto tokens:
ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00-0.40, cash 0.00-0.15 (aggressive stance), sum=1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose. Start with the opening brace.
Include fields: portfolio_weights (dict), cash_weight (float), bull_thesis (string, ≤80 chars)."""

BEAR_RESEARCHER_SYSTEM = """\
You are the Bear Researcher in a TradingAgents multi-agent trading system.
Your role is to synthesize analyst reports and construct the strongest BEARISH
/ risk-off case, proposing a defensive portfolio.
Respond ONLY in valid JSON. No prose, no markdown."""

BEAR_RESEARCHER_USER = """\
=== ANALYST REPORTS ===

[TECHNICAL ANALYST]:
{tech_report}

[SENTIMENT & ON-CHAIN ANALYST]:
{sentiment_report}

[MACRO ANALYST]:
{macro_report}

As the Bear Researcher, identify the strongest bearish/risk signals and propose a
defensive portfolio allocation across 13 crypto tokens:
ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00-0.40, cash 0.10-0.30 (defensive stance), sum=1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose. Start with the opening brace.
Include fields: portfolio_weights (dict), cash_weight (float), bear_thesis (string, ≤80 chars)."""

RISK_MANAGER_SYSTEM = """\
You are the Risk Manager in a TradingAgents multi-agent trading system.
You receive proposals from a Bull Researcher (optimistic) and a Bear Researcher
(defensive) and must produce the final balanced portfolio allocation, weighing
both cases based on evidence strength.
Respond ONLY in valid JSON. No prose, no markdown."""

RISK_MANAGER_USER = """\
=== BULL RESEARCHER PROPOSAL ===
Thesis: {bull_thesis}
Weights (top tokens): {bull_top5}
Cash: {bull_cash:.1%}

=== BEAR RESEARCHER PROPOSAL ===
Thesis: {bear_thesis}
Weights (top tokens): {bear_top5}
Cash: {bear_cash:.1%}

=== YOUR TASK ===
As Risk Manager, weigh the bull vs bear evidence and produce the final portfolio
for 13 crypto tokens: ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00-0.40, cash 0.00-0.30, all weights + cash = 1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose. Start with the opening brace.
Include fields: portfolio_weights (dict with all 13 tokens), cash_weight (float),
market_regime (string: bullish/bearish/neutral/mixed), risk_level (string: low/medium/high)."""


# ---------------------------------------------------------------------------
#  TradingAgentsCouncil
# ---------------------------------------------------------------------------

class TradingAgentsCouncil:
    """
    TradingAgents (2025) multi-agent baseline for crypto portfolio allocation.

    See module docstring for architecture details.
    """

    def __init__(self, llm: BaseLLM, lookback_days: int = 90):
        self.llm           = llm
        self.lookback_days = lookback_days

        # Stage 1 analyst agents
        self.agent1 = MarketObserver(llm=llm)
        self.agent2 = OnchainObserver(llm=llm)
        self.agent3 = EnvironmentObserver(llm=llm)

        # Utility
        self._analyst = InvestmentAnalyst(llm=llm)

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        logger.info("[TradingAgents] run_with_bundle: eval_date=%s", bundle.eval_date)

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
            "[TradingAgents] Stage 1 done | conf1=%.2f conf2=%.2f conf3=%.2f",
            eval1.confidence, eval2.confidence, eval3.confidence,
        )

        # Stage 2: adversarial bull / bear debate (2 LLM calls)
        bull_parsed, bear_parsed = await self._run_debate(eval1, eval2, eval3)

        # Stage 3: risk manager final decision (1 LLM call)
        weights, cash_weight = await self._risk_manager_decision(
            bull_parsed, bear_parsed
        )

        logger.info("[TradingAgents] Risk manager done | cash_weight=%.2f", cash_weight)

        return {
            "portfolio_weights": weights,
            "cash_weight":       cash_weight,
            "market_regime":     "trading_agents_decision",
        }

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        pass

    # ------------------------------------------------------------------
    #  Stage 2: Bull / Bear debate
    # ------------------------------------------------------------------

    async def _run_debate(self, eval1, eval2, eval3):
        """Run Bull and Bear researchers (can be parallelised on multi-GPU; sequential here)."""
        tech_compact = InvestmentAnalyst._compact_stage1(eval1, max_raw=350)
        sent_compact = InvestmentAnalyst._compact_stage1(eval2, max_raw=350)
        macro_compact = InvestmentAnalyst._compact_stage1(eval3, max_raw=350)

        # Bull researcher
        bull_resp = await self.llm.chat_text(
            system_prompt=BULL_RESEARCHER_SYSTEM,
            user_prompt=BULL_RESEARCHER_USER.format(
                tech_report      = tech_compact,
                sentiment_report = sent_compact,
                macro_report     = macro_compact,
            ),
        )
        bull_parsed = self._analyst._parse_json(bull_resp)

        # Bear researcher
        bear_resp = await self.llm.chat_text(
            system_prompt=BEAR_RESEARCHER_SYSTEM,
            user_prompt=BEAR_RESEARCHER_USER.format(
                tech_report      = tech_compact,
                sentiment_report = sent_compact,
                macro_report     = macro_compact,
            ),
        )
        bear_parsed = self._analyst._parse_json(bear_resp)

        return bull_parsed, bear_parsed

    # ------------------------------------------------------------------
    #  Stage 3: Risk Manager
    # ------------------------------------------------------------------

    async def _risk_manager_decision(
        self,
        bull_parsed: Dict[str, Any],
        bear_parsed: Dict[str, Any],
    ) -> Tuple[Dict[str, float], float]:
        """Risk Manager balances bull and bear proposals → final allocation."""

        def _top5_str(weights: Dict[str, float]) -> str:
            if not weights:
                return "N/A"
            top5 = sorted(weights.items(), key=lambda x: -float(x[1]))[:5]
            return ", ".join(f"{k}={float(v):.3f}" for k, v in top5)

        bull_weights = bull_parsed.get("portfolio_weights", {})
        bear_weights = bear_parsed.get("portfolio_weights", {})
        bull_cash    = float(bull_parsed.get("cash_weight", 0.05))
        bear_cash    = float(bear_parsed.get("cash_weight", 0.20))

        response = await self.llm.chat_text(
            system_prompt=RISK_MANAGER_SYSTEM,
            user_prompt=RISK_MANAGER_USER.format(
                bull_thesis = bull_parsed.get("bull_thesis", "Bullish market conditions"),
                bull_top5   = _top5_str(bull_weights),
                bull_cash   = bull_cash,
                bear_thesis = bear_parsed.get("bear_thesis", "Bearish risk conditions"),
                bear_top5   = _top5_str(bear_weights),
                bear_cash   = bear_cash,
            ),
        )

        parsed = self._analyst._parse_json(response)

        norm = self._analyst._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )
        return norm["token_weights"], norm["cash_weight"]
