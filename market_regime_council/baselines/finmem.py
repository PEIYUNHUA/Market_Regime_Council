"""
FinMem Baseline (AAAI Spring Symposium 2024 / ICLR Workshop 2024)

Reference:
  "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory
   and Character Design"
  Yangyang Yu et al., 2024  |  GitHub: pipiku915/FinMem-LLM-StockTrading

Architecture reproduced here (adapted for crypto portfolio allocation):
  The agent maintains three memory layers that persist across periods:
    · Working Memory    — Current period data summary (reset each period)
    · Short-Term Memory — Last K (default K=5) period decisions + outcomes
    · Long-Term Memory  — Distilled patterns from all history (LLM-updated summary)

  Each period:
  1. Build working memory from current DateRangeBundle (via existing formatters)
  2. Retrieve short-term memory (last 5 allocations + their realized returns)
  3. Query long-term memory (LLM summary of historical patterns)
  4. Single LLM agent call with all three memory layers → portfolio weights
  5. Update long-term memory summary (LLM call every K periods or on request)

Key difference from MRC: single agent with explicit cross-period memory instead
of multi-agent Shapley-weighted synthesis.

LLM call count per period: 3 (analyst formatters) + 1 (FinMem agent) = 4 total.
Long-term memory refresh: 1 extra LLM call every 5 periods.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

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

FINMEM_AGENT_SYSTEM = """\
You are a FinMem crypto portfolio agent with a layered memory architecture.
You have access to three memory layers:
  - Working Memory:   current market data (this period)
  - Short-Term Memory: your last 5 allocation decisions and their outcomes
  - Long-Term Memory:  distilled historical patterns across all periods
Use all three memory layers to make an informed portfolio allocation decision.
Respond ONLY in valid JSON. No analysis text, no markdown, no prose."""

FINMEM_AGENT_USER = """\
=== WORKING MEMORY (current period data) ===
{working_memory}

=== SHORT-TERM MEMORY (last {n_short_term} periods) ===
{short_term_memory}

=== LONG-TERM MEMORY (historical patterns) ===
{long_term_memory}

Based on all three memory layers, allocate a portfolio across these 13 crypto tokens:
ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC

Portfolio constraints: each token 0.00-0.40, cash 0.00-0.30, sum of all = 1.00.

IMPORTANT: Your response must be ONLY a valid JSON object. No prose. Start with the opening brace.
Include fields: portfolio_weights (dict, all 13 tokens), cash_weight (float),
market_regime (string), risk_level (string), memory_note (string, ≤80 chars,
brief reflection on what this period taught you)."""

FINMEM_MEMORY_UPDATE_SYSTEM = """\
You are a memory consolidation module for a FinMem crypto trading agent.
Distill key patterns and lessons from recent trading history into a concise
long-term memory summary (≤200 words) that will guide future decisions.
Respond ONLY in valid JSON with field: long_term_memory (string)."""

FINMEM_MEMORY_UPDATE_USER = """\
=== EXISTING LONG-TERM MEMORY ===
{existing_ltm}

=== RECENT SHORT-TERM MEMORIES TO INTEGRATE ===
{recent_stm}

Update the long-term memory to incorporate lessons from recent periods.
Keep the summary concise (≤200 words) and actionable.

IMPORTANT: Respond ONLY with valid JSON containing field: long_term_memory (string)."""

# ---------------------------------------------------------------------------
#  FinMemCouncil
# ---------------------------------------------------------------------------

class FinMemCouncil:
    """
    FinMem (AAAI 2024) layered-memory single-agent baseline.

    See module docstring for architecture details.
    """

    SHORT_TERM_K = 5        # number of periods in short-term memory
    LTM_REFRESH_EVERY = 5   # refresh long-term memory every N periods

    def __init__(self, llm: BaseLLM, lookback_days: int = 90):
        self.llm           = llm
        self.lookback_days = lookback_days

        # Analyst agents (for data formatting via their analyze() methods)
        self.agent1 = MarketObserver(llm=llm)
        self.agent2 = OnchainObserver(llm=llm)
        self.agent3 = EnvironmentObserver(llm=llm)

        # Utility
        self._analyst = InvestmentAnalyst(llm=llm)

        # Memory layers
        self._short_term: List[Dict[str, Any]] = []   # list of dicts
        self._long_term: str = (
            "No historical data yet. This is the first allocation period. "
            "Begin with a balanced, diversified portfolio and observe market conditions."
        )
        self._period_count: int = 0

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        logger.info("[FinMem] run_with_bundle: eval_date=%s", bundle.eval_date)

        # Obtain analyst evaluations for working memory (3 LLM calls)
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
            "[FinMem] Analyst analyses done | conf1=%.2f conf2=%.2f conf3=%.2f",
            eval1.confidence, eval2.confidence, eval3.confidence,
        )

        # Periodically refresh long-term memory
        if self._period_count > 0 and self._period_count % self.LTM_REFRESH_EVERY == 0:
            await self._refresh_long_term_memory()

        # Build memory representations
        working_memory   = self._build_working_memory(eval1, eval2, eval3)
        short_term_text  = self._format_short_term()
        long_term_text   = self._long_term

        # FinMem agent decision (1 LLM call)
        response = await self.llm.chat_text(
            system_prompt=FINMEM_AGENT_SYSTEM,
            user_prompt=FINMEM_AGENT_USER.format(
                working_memory   = working_memory,
                n_short_term     = len(self._short_term),
                short_term_memory = short_term_text,
                long_term_memory = long_term_text,
            ),
        )

        parsed = self._analyst._parse_json(response)
        norm = self._analyst._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )
        weights    = norm["token_weights"]
        cash_weight = norm["cash_weight"]

        # Store in short-term memory (actual_return will be filled in record_period_result)
        self._short_term.append({
            "eval_date":     str(bundle.eval_date),
            "weights":       weights,
            "cash_weight":   cash_weight,
            "market_regime": parsed.get("market_regime", "unknown"),
            "memory_note":   parsed.get("memory_note", ""),
            "actual_return": None,  # filled later via record_period_result
        })
        if len(self._short_term) > self.SHORT_TERM_K:
            self._short_term.pop(0)

        self._period_count += 1
        logger.info("[FinMem] Decision done | cash_weight=%.2f period=%d",
                    cash_weight, self._period_count)

        return {
            "portfolio_weights": weights,
            "cash_weight":       cash_weight,
            "market_regime":     parsed.get("market_regime", "finmem_decision"),
        }

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        """Back-fill the realized return into the most recent short-term memory entry."""
        if self._short_term and self._short_term[-1].get("actual_return") is None:
            self._short_term[-1]["actual_return"] = round(actual_return, 6)

    # ------------------------------------------------------------------
    #  Memory management
    # ------------------------------------------------------------------

    def _build_working_memory(self, eval1, eval2, eval3) -> str:
        """Combine current analyst evaluations into a working memory string."""
        parts = []
        tech  = InvestmentAnalyst._compact_stage1(eval1, max_raw=300)
        sent  = InvestmentAnalyst._compact_stage1(eval2, max_raw=300)
        macro = InvestmentAnalyst._compact_stage1(eval3, max_raw=300)
        parts.append(f"[Technical]\n{tech}")
        parts.append(f"[Sentiment/On-chain]\n{sent}")
        parts.append(f"[Macro]\n{macro}")
        return "\n\n".join(parts)

    def _format_short_term(self) -> str:
        """Format short-term memory as a concise list of recent decisions."""
        if not self._short_term:
            return "No previous decisions recorded."
        lines = []
        for i, entry in enumerate(self._short_term, 1):
            ret_str = (
                f"{entry['actual_return']:+.2%}"
                if entry.get("actual_return") is not None
                else "pending"
            )
            # Top 3 weights
            w = entry.get("weights", {})
            top3 = sorted(w.items(), key=lambda x: -float(x[1]))[:3]
            top3_str = ", ".join(f"{k}={float(v):.3f}" for k, v in top3)
            lines.append(
                f"Period -{len(self._short_term)-i+1}: "
                f"date={entry['eval_date']} regime={entry['market_regime']} "
                f"cash={entry['cash_weight']:.2f} top3=[{top3_str}] "
                f"realized_return={ret_str} | note: {entry.get('memory_note', '')}"
            )
        return "\n".join(lines)

    async def _refresh_long_term_memory(self):
        """Periodically distil short-term memories into long-term memory via LLM."""
        if not self._short_term:
            return
        stm_text = self._format_short_term()
        try:
            response = await self.llm.chat_text(
                system_prompt=FINMEM_MEMORY_UPDATE_SYSTEM,
                user_prompt=FINMEM_MEMORY_UPDATE_USER.format(
                    existing_ltm = self._long_term,
                    recent_stm   = stm_text,
                ),
            )
            parsed = self._analyst._parse_json(response)
            new_ltm = parsed.get("long_term_memory", "")
            if new_ltm:
                self._long_term = new_ltm
                logger.info("[FinMem] Long-term memory refreshed (period %d)", self._period_count)
        except Exception as e:
            logger.warning("[FinMem] LTM refresh failed (%s); keeping existing LTM", e)
