"""
FS-ReasoningAgent Baseline (ICLR 2025 Workshop)

Reference:
  "Exploring LLM Cryptocurrency Trading Through Fact-Subjectivity Aware Reasoning"
  ICLR 2025 Advances in Financial AI Workshop
  arXiv: https://arxiv.org/abs/2410.12464
  GitHub: Persdre/FS-ReasoningAgent

Architecture reproduced here (adapted for 13-token portfolio allocation):
  Two parallel LLM agents that reason over orthogonal information streams:

    · Fact Agent         — reasons ONLY from objective, verifiable data:
                           OHLCV price statistics, on-chain blockchain metrics
                           (AdrActCnt, TxCnt, ChainTVL), FRED macro indicators
                           (VIX, DXY, US10Y, CPI, etc.)
    · Subjectivity Agent — reasons ONLY from subjective, sentiment-driven data:
                           Polymarket prediction market probabilities,
                           Fear & Greed Index trend, qualitative narrative signals,
                           general market mood

  Both agents receive ALL analyst data but are explicitly instructed to filter
  their reasoning to their assigned information type. This separation is the
  paper's key contribution: it avoids the confirmation bias that arises when
  a single agent blends objective signals with subjective narratives.

  Final portfolio: equal-weight blend of both agents' softmax-normalised
  portfolio outputs (weights averaged token-by-token, then re-normalised).

Key difference from MRC: the two-stream separation is static and symmetric —
there is no Shapley-based dynamic reweighting, no cross-period learning, and
no pairwise Socratic debate. The paper's finding: separating fact from
subjectivity consistently improves Sharpe vs. single-agent baselines.

LLM calls per period:
  3 (analyst formatters) + 2 (fact agent || subjectivity agent) = 5 total.
  Fact and Subjectivity agents run concurrently (asyncio.gather).
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

FACT_AGENT_SYSTEM = """\
You are the Fact Agent in an FS-ReasoningAgent crypto trading system.
Your role is to reason EXCLUSIVELY from objective, verifiable, numerical data:
  - Price statistics (OHLCV returns, momentum, volatility)
  - On-chain blockchain metrics (active addresses, transaction counts, TVL)
  - Macroeconomic indicators (VIX, DXY, yield curve, CPI, unemployment)
Do NOT let market sentiment, news narratives, or fear/greed indices influence
your reasoning. Base your allocation purely on measurable facts.
Respond ONLY in valid JSON. No analysis text, no markdown, no prose."""

FACT_AGENT_USER = """\
=== OBJECTIVE DATA (for factual reasoning only) ===

[Price & Momentum Statistics (OHLCV, technical indicators)]:
{tech_report}

[On-Chain Blockchain Metrics (AdrActCnt, TxCnt, ChainTVL, fees)]:
{onchain_report}

[Macroeconomic Indicators (FRED: VIX, DXY, US10Y, CPI, unemployment)]:
{macro_report}

Reason ONLY from the objective metrics above. Ignore any sentiment or narrative
language. Allocate a portfolio across these 13 crypto tokens:
ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00–0.40, cash 0.00–0.30, sum of all = 1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose, no markdown.
Include fields: portfolio_weights (dict, all 13 tokens), cash_weight (float),
market_regime (string: "bull" | "volatile" | "bear"),
reasoning_basis (string, ≤80 chars: which objective metrics drove your decision)."""

SUBJECTIVITY_AGENT_SYSTEM = """\
You are the Subjectivity Agent in an FS-ReasoningAgent crypto trading system.
Your role is to reason EXCLUSIVELY from subjective, sentiment-driven signals:
  - Polymarket prediction market probabilities (crowd wisdom on price outcomes)
  - Fear & Greed Index trend (market psychology, 30-day window)
  - Qualitative market narrative and mood signals
Do NOT let price charts, on-chain metrics, or macro indicators influence your
reasoning. Base your allocation purely on what market participants believe and feel.
Respond ONLY in valid JSON. No analysis text, no markdown, no prose."""

SUBJECTIVITY_AGENT_USER = """\
=== SUBJECTIVE DATA (for sentiment reasoning only) ===

[Polymarket Prediction Market Sentiment]:
{sentiment_report}

[Fear & Greed Index & Macro Sentiment Narrative]:
{macro_report}

[Market Regime Signals (qualitative context from all data sources)]:
Regime signals — Tech: {regime1} | On-chain: {regime2} | Macro: {regime3}

Reason ONLY from the subjective sentiment signals above. Ignore raw price
statistics, on-chain counts, or macro indicator values. Allocate a portfolio
based on what the crowd and market mood are signalling about upcoming moves.
Tokens: ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00–0.40, cash 0.00–0.30, sum of all = 1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose, no markdown.
Include fields: portfolio_weights (dict, all 13 tokens), cash_weight (float),
market_regime (string: "bull" | "volatile" | "bear"),
sentiment_signal (string, ≤80 chars: which sentiment signals drove your decision)."""

# ---------------------------------------------------------------------------
#  FSReasoningAgentCouncil
# ---------------------------------------------------------------------------

class FSReasoningAgentCouncil:
    """
    FS-ReasoningAgent (ICLR 2025 Workshop) two-stream baseline.

    See module docstring for architecture details.
    """

    def __init__(self, llm: BaseLLM, lookback_days: int = 90):
        self.llm           = llm
        self.lookback_days = lookback_days

        # Analyst agents (same data pipeline as all other baselines)
        self.agent1 = MarketObserver(llm=llm)
        self.agent2 = OnchainObserver(llm=llm)
        self.agent3 = EnvironmentObserver(llm=llm)

        self._analyst = InvestmentAnalyst(llm=llm)

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        logger.info("[FSReasoning] run_with_bundle: eval_date=%s", bundle.eval_date)

        # Stage 1: three analyst analyses (3 LLM calls, same as all baselines)
        eval1, eval2, eval3 = await asyncio.gather(
            self.agent1.analyze(
                bundle.market_snapshots,
                kline_images=bundle.kline_images,
                lookback_days=self.lookback_days,
            ),
            self.agent2.analyze(
                bundle.onchain_metrics,
                btc_macro_df=bundle.btc_macro_df,
            ),
            self.agent3.analyze(
                bundle.env_snapshot,
                fred_df=bundle.fred_df,
                fng_df=bundle.fng_df,
                tweets_df=bundle.tweets_df,
                polymarket_df=bundle.polymarket_df,
            ),
        )

        regime1 = eval1.parsed.get("market_regime", "unknown")
        regime2 = eval2.parsed.get("market_regime", "unknown")
        regime3 = eval3.parsed.get("market_regime", "unknown")
        logger.info(
            "[FSReasoning] Stage 1 done | regime1=%s regime2=%s regime3=%s",
            regime1, regime2, regime3,
        )

        tech_compact  = InvestmentAnalyst._compact_stage1(eval1, max_raw=350)
        sent_compact  = InvestmentAnalyst._compact_stage1(eval2, max_raw=350)
        macro_compact = InvestmentAnalyst._compact_stage1(eval3, max_raw=350)

        # Stage 2: Fact Agent and Subjectivity Agent run concurrently (2 LLM calls)
        fact_w, fact_cash, sub_w, sub_cash = await self._run_two_agents(
            tech_compact, sent_compact, macro_compact,
            regime1, regime2, regime3,
        )

        # Blend: equal-weight average of both agents' outputs
        weights, cash_weight = self._blend(fact_w, fact_cash, sub_w, sub_cash)

        logger.info(
            "[FSReasoning] Blended | fact_cash=%.2f sub_cash=%.2f final_cash=%.2f",
            fact_cash, sub_cash, cash_weight,
        )

        return {
            "portfolio_weights": weights,
            "cash_weight":       cash_weight,
            "market_regime":     "fs_reasoning_decision",
        }

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        """No cross-period state in FS-ReasoningAgent (stateless by design)."""
        pass

    # ------------------------------------------------------------------
    #  Two-stream agents
    # ------------------------------------------------------------------

    async def _run_two_agents(
        self,
        tech_compact: str,
        sent_compact: str,
        macro_compact: str,
        regime1: str,
        regime2: str,
        regime3: str,
    ) -> Tuple[Dict[str, float], float, Dict[str, float], float]:
        """Run Fact Agent and Subjectivity Agent concurrently."""

        fact_resp, sub_resp = await asyncio.gather(
            self.llm.chat_text(
                system_prompt=FACT_AGENT_SYSTEM,
                user_prompt=FACT_AGENT_USER.format(
                    tech_report    = tech_compact,
                    onchain_report = sent_compact,
                    macro_report   = macro_compact,
                ),
            ),
            self.llm.chat_text(
                system_prompt=SUBJECTIVITY_AGENT_SYSTEM,
                user_prompt=SUBJECTIVITY_AGENT_USER.format(
                    sentiment_report = sent_compact,
                    macro_report     = macro_compact,
                    regime1          = regime1,
                    regime2          = regime2,
                    regime3          = regime3,
                ),
            ),
        )

        fact_parsed = self._analyst._parse_json(fact_resp)
        sub_parsed  = self._analyst._parse_json(sub_resp)

        fact_norm = self._analyst._normalize_weights(
            fact_parsed.get("portfolio_weights", {}),
            fact_parsed.get("cash_weight", 0.0),
        )
        sub_norm = self._analyst._normalize_weights(
            sub_parsed.get("portfolio_weights", {}),
            sub_parsed.get("cash_weight", 0.0),
        )

        logger.debug(
            "[FSReasoning] Fact: cash=%.2f | Sub: cash=%.2f",
            fact_norm["cash_weight"], sub_norm["cash_weight"],
        )

        return (
            fact_norm["token_weights"], fact_norm["cash_weight"],
            sub_norm["token_weights"],  sub_norm["cash_weight"],
        )

    # ------------------------------------------------------------------
    #  Equal-weight blending
    # ------------------------------------------------------------------

    def _blend(
        self,
        fact_w: Dict[str, float],
        fact_cash: float,
        sub_w: Dict[str, float],
        sub_cash: float,
    ) -> Tuple[Dict[str, float], float]:
        """
        Equal-weight average of both agents' token weights and cash weights,
        then re-normalise via InvestmentAnalyst._normalize_weights.
        """
        all_tokens = set(list(fact_w.keys()) + list(sub_w.keys()))
        blended: Dict[str, float] = {
            tok: (fact_w.get(tok, 0.0) + sub_w.get(tok, 0.0)) / 2.0
            for tok in all_tokens
        }
        blended_cash = (fact_cash + sub_cash) / 2.0

        norm = self._analyst._normalize_weights(blended, blended_cash)
        return norm["token_weights"], norm["cash_weight"]
