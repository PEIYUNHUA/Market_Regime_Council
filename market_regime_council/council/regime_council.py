""""""
from __future__ import annotations

import datetime
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..agents.base_agent import AgentEvaluation
from ..agents.market_observer import MarketObserver
from ..agents.onchain_observer import OnchainObserver
from ..agents.environment_observer import EnvironmentObserver
from ..agents.investment_analyst import InvestmentAnalyst
from ..data_layer.data_manager import DataManager
from ..data_layer.base import TokenSnapshot
from ..llm.base import BaseLLM
from ..prompts.templates import (
    COORDINATOR_SYSTEM, COORDINATOR_USER,
    PAIRWISE_DEBATE_SYSTEM, PAIRWISE_DEBATE_USER,
)
from ..config import AgentConfig, ALL_TOKENS
from .shapley_tracker import ShapleyValueLedger, ShapleyStageRecord, STAGE2_PAIRS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

#


#


# ---------------------------------------------------------------------------


_DOMINANCE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "technical": {"agent1": 1.30, "agent2": 0.90, "agent3": 0.85},
    "onchain":   {"agent1": 0.85, "agent2": 1.30, "agent3": 0.90},
    "macro":     {"agent1": 0.85, "agent2": 0.90, "agent3": 1.30},
    "other":     {"agent1": 1.0,  "agent2": 1.0,  "agent3": 1.0},

}


# ---------------------------------------------------------------------------
# Performance Evaluator
# ---------------------------------------------------------------------------

@dataclass
class PerformanceRecord:
    date: datetime.date
    prediction: Dict[str, Any]
    actual_return: float = 0.0
    portfolio_value: float = 0.0


class PerformanceEvaluator:
    """"""

    def __init__(self):
        self._records: List[PerformanceRecord] = []

    def record(self, date: datetime.date, prediction: Dict[str, Any],
               actual_return: float = 0.0, portfolio_value: float = 0.0):
        self._records.append(PerformanceRecord(
            date=date, prediction=prediction,
            actual_return=actual_return, portfolio_value=portfolio_value,
        ))

    def get_feedback_summary(self, lookback: int = 12) -> str:
        """"""
        if len(self._records) < 2:
            return "No performance history yet. This is the first allocation decision."

        recent = self._records[-lookback:]
        returns = [r.actual_return for r in recent if r.actual_return != 0]

        if len(returns) < 2:
            return f"Insufficient history ({len(recent)} records). Proceed with equal confidence."


        mean_ret = sum(returns) / len(returns)
        std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1))
        sharpe = (mean_ret / std_ret * math.sqrt(52)) if std_ret > 0 else 0.0

        downside = [r for r in returns if r < 0]
        std_down = math.sqrt(sum(r ** 2 for r in downside) / len(downside)) if downside else 0.0
        sortino = (mean_ret / std_down * math.sqrt(52)) if std_down > 0 else sharpe

        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        # MDD on recent records
        values = [r.portfolio_value for r in recent if r.portfolio_value > 0]
        mdd = self._max_drawdown(values)

        return (
            f"Recent {len(returns)}-week performance: "
            f"Sharpe={sharpe:.2f} | Sortino={sortino:.2f} | "
            f"Win Rate={win_rate:.1%} | MDD={mdd:.1%} | "
            f"Avg Weekly Return={mean_ret:.2%}"
        )

    def get_recent_correctness(self, lookback: int = 10) -> Dict[str, float]:
        """"""
        recent = self._records[-lookback:] if self._records else []
        if not recent:
            return {"agent1": 0.5, "agent2": 0.5, "agent3": 0.5}
        win_rate = sum(1 for r in recent if r.actual_return > 0) / len(recent)
        return {"agent1": win_rate, "agent2": win_rate, "agent3": win_rate}

    def get_summary(self) -> str:
        if not self._records:
            return "No performance records yet."
        total = len(self._records)
        profitable = sum(1 for r in self._records if r.actual_return > 0)
        return (
            f"Total predictions: {total} | Profitable: {profitable}/{total} | "
            f"Feedback: {self.get_feedback_summary()}"
        )

    @staticmethod
    def _max_drawdown(values: List[float]) -> float:
        if not values:
            return 0.0
        peak = values[0]
        mdd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            mdd = max(mdd, dd)
        return mdd


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:

    def __init__(self, llm: BaseLLM, max_rounds: int = 3):
        self.llm = llm
        self.max_rounds = max_rounds

    async def facilitate_discussion(
        self,
        agent1_eval: AgentEvaluation,
        agent2_eval: AgentEvaluation,
        agent3_eval: AgentEvaluation,
        agents: Dict[str, Any] = None,
    ) -> Dict[str, AgentEvaluation]:
        """"""
        current_evals = {"agent1": agent1_eval, "agent2": agent2_eval, "agent3": agent3_eval}
        discussion_history = ""

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"=== Discussion Round {round_num}/{self.max_rounds} ===")

            user_prompt = COORDINATOR_USER.format(
                agent1_eval=current_evals["agent1"].raw_text,
                agent2_eval=current_evals["agent2"].raw_text,
                agent3_eval=current_evals["agent3"].raw_text,
                discussion_history=discussion_history,
                round_num=round_num,
            )
            coord_response = await self.llm.chat_text(
                system_prompt=COORDINATOR_SYSTEM, user_prompt=user_prompt,
            )
            discussion_history += f"\n--- Round {round_num} ---\n{coord_response}\n"

            if agents and round_num < self.max_rounds:
                for key, agent in agents.items():
                    others = [v for k, v in current_evals.items() if k != key]
                    revised = await agent.discuss(current_evals[key], others)
                    current_evals[key] = revised

            try:
                coord_parsed = json.loads(coord_response)
                if coord_parsed.get("consensus_reached", False):
                    logger.info("Consensus reached, ending discussion.")
                    break
            except json.JSONDecodeError:
                pass

        return current_evals


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class DateRangeBundle:
    """"""
    eval_date: datetime.date

    market_snapshots: Dict[str, TokenSnapshot] = field(default_factory=dict)
    kline_images: Dict[str, str] = field(default_factory=dict)

    onchain_metrics: Dict = field(default_factory=dict)             # Dict[str, OnchainMetrics]
    btc_macro_df: Optional[pd.DataFrame] = None
    polymarket_df: Optional[pd.DataFrame] = None
    tweets_df: Optional[pd.DataFrame] = None

    env_snapshot: Any = None                                         # EnvironmentSnapshot
    fred_df: Optional[pd.DataFrame] = None
    fng_df: Optional[pd.DataFrame] = None

    prices: Dict[str, float] = field(default_factory=dict)

    market_regime: Optional[str] = None

    regime_score: float = 0.0

    regime_transition_boost: Optional[Dict[str, float]] = None


    btc_dominance: float = 0.0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class MarketRegimeDetector:
    """"""

    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"


    _SCORE_BULL = 0.30
    _SCORE_BEAR = -0.30

    def detect(
        self,
        btc_price_now: float,
        btc_price_30d_ago: Optional[float],
        btc_price_7d_ago: Optional[float] = None,
        vol_monthly: Optional[float] = None,
    ) -> Tuple[str, float]:
        """"""
        if not btc_price_30d_ago or btc_price_30d_ago <= 0 or btc_price_now <= 0:
            return self.VOLATILE, 0.0

        mom_30d = btc_price_now / btc_price_30d_ago - 1.0


        ref_vol = max(vol_monthly, 0.05) if vol_monthly is not None else 0.10


        regime_score = math.tanh(mom_30d / ref_vol)


        if btc_price_7d_ago and btc_price_7d_ago > 0:
            mom_7d = btc_price_now / btc_price_7d_ago - 1.0
            if mom_30d * mom_7d < 0 and abs(mom_7d) > 0.30 * abs(mom_30d):
                regime_score *= 0.5


        if regime_score > self._SCORE_BULL:
            regime = self.BULL
        elif regime_score < self._SCORE_BEAR:
            regime = self.BEAR
        else:
            regime = self.VOLATILE

        return regime, regime_score

    @staticmethod
    def get_multipliers(regime: str) -> Dict[str, float]:
        """"""
        return _REGIME_MULTIPLIERS.get(regime, {"agent1": 1.0, "agent2": 1.0, "agent3": 1.0})

    @staticmethod
    def get_soft_multipliers(
        regime_score: float,
        transition_boost: Optional[Dict[str, float]] = None,
        bull_mults:    Tuple[float, float, float] = (1.5, 0.9, 0.6),
        neutral_mults: Tuple[float, float, float] = (1.2, 1.0, 0.8),
        bear_mults:    Tuple[float, float, float] = (0.6, 0.9, 1.5),
    ) -> Dict[str, float]:
        """"""

        bull    = max(0.0, regime_score)
        bear    = max(0.0, -regime_score)
        neutral = 1.0 - abs(regime_score)


        a1 = bull_mults[0] * bull + neutral_mults[0] * neutral + bear_mults[0] * bear
        a2 = bull_mults[1] * bull + neutral_mults[1] * neutral + bear_mults[1] * bear
        a3 = bull_mults[2] * bull + neutral_mults[2] * neutral + bear_mults[2] * bear

        mults = {"agent1": a1, "agent2": a2, "agent3": a3}


        if transition_boost:
            for agent_id, boost_factor in transition_boost.items():
                if agent_id in mults:
                    mults[agent_id] = mults[agent_id] * boost_factor

        return mults


# ===========================================================================

# ===========================================================================

class RegimeCouncil:
    """"""

    def __init__(
        self,
        llm: BaseLLM,
        data_manager: DataManager = None,
        config: AgentConfig = None,
        lookback_days: int = 90,
        risk_free_rate: float = 0.0,
        freq: int = 365,
        softmax_temperature: float = 0.20,


        agent4_blend: Optional[float] = None,
        s1_s2_blend:  Optional[float] = None,


        regime_bull_mults:    Tuple[float, float, float] = (1.5, 0.9, 0.6),
        regime_neutral_mults: Tuple[float, float, float] = (1.2, 1.0, 0.8),


        regime_bear_mults:    Tuple[float, float, float] = (0.6, 1.3, 1.1),

        sharpe_weight:          float = 0.4,
        return_weight:          float = 0.6,
        pair_quality_threshold: float = 0.40,
        ew_halflife:            Optional[int] = 252,


        agent_prior: Optional[Dict[str, float]] = None,


        pair_prior:  Optional[Dict[str, float]] = None,

        allow_short:     bool  = False,
        max_short_weight: float = 0.20,
    ):
        self.config        = config or AgentConfig()
        self.data_manager  = data_manager
        self.lookback_days = lookback_days
        self._llm          = llm


        self.agent1 = MarketObserver(llm=llm)
        self.agent2 = OnchainObserver(llm=llm)
        self.agent3 = EnvironmentObserver(llm=llm)
        self.agent4 = InvestmentAnalyst(
            llm=llm,
            allow_short=allow_short,
            max_short=max_short_weight,
        )


        self.shapley_ledger = ShapleyValueLedger(
            risk_free_rate         = risk_free_rate,
            freq                   = freq,
            softmax_temperature    = softmax_temperature,
            sharpe_weight          = sharpe_weight,
            return_weight          = return_weight,
            pair_quality_threshold = pair_quality_threshold,
            ew_halflife            = ew_halflife,
            agent_prior            = agent_prior,
            pair_prior             = pair_prior,
        )


        self.regime_detector = MarketRegimeDetector()


        self._regime_bull_mults    = regime_bull_mults
        self._regime_neutral_mults = regime_neutral_mults
        self._regime_bear_mults    = regime_bear_mults


        self.performance_eval = PerformanceEvaluator()
        self.current_weights  = dict(self.config.initial_weights)


        self._agent4_blend_fixed: Optional[float] = agent4_blend
        self._s1_s2_blend_fixed:  Optional[float] = s1_s2_blend

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _compute_blend_ratios(
        self,
        agent_weights: Dict[str, float],
        pair_weights:  Dict[str, float],
    ) -> Tuple[float, float]:
        """"""
        ledger   = self.shapley_ledger
        fixed_s1 = self._s1_s2_blend_fixed
        fixed_a4 = self._agent4_blend_fixed

        if not ledger._has_enough_history():
            return (
                fixed_s1 if fixed_s1 is not None else 0.90,
                fixed_a4 if fixed_a4 is not None else 0.10,
            )


        s1_ens = sum(
            agent_weights.get(a, 1 / 3) * ledger._coalition_value(ledger._s1_ret[a])
            for a in ["agent1", "agent2", "agent3"]
        )
        s1_ens = max(s1_ens, 1e-6)


        if fixed_s1 is not None:
            s1_blend = fixed_s1
        else:
            s2_ens = sum(
                pair_weights.get(p, 1 / 3) * ledger._coalition_value(ledger._s2_ret[p])
                for p in ["1_2", "1_3", "2_3"]
            )
            delta_s2 = s2_ens - s1_ens
            s1_blend = 0.90 + 0.09 * math.tanh(-delta_s2 / 0.075)


        if fixed_a4 is not None:
            agent4_blend = fixed_a4
        else:
            s3       = ledger._coalition_value(ledger._s3_ret)
            delta_a4 = s3 - s1_ens
            agent4_blend = max(0.0, 0.15 + 0.20 * math.tanh(delta_a4 / 0.10))

        logger.debug(
            "[ShapleyCouncil] Coalition-value-driven blends — "
            f"s1_blend={s1_blend:.3f}  agent4_blend={agent4_blend:.3f}  "
            f"(s1_ens={s1_ens:.3f})"
        )
        return s1_blend, agent4_blend

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        """"""
        logger.info(f"[ShapleyCouncil] run_with_bundle: eval_date={bundle.eval_date}")


        logger.info("[ShapleyCouncil] Stage 1: Private independent predictions")


        btc_dom_val = getattr(bundle, "btc_dominance", 0.0)
        if abs(btc_dom_val) > 0.02:
            _dom_label = (
                "BTC season (BTC outperforming alts)"
                if btc_dom_val > 0
                else "Alt season (alts outperforming BTC)"
            )
            _dom_hint = (
                "Increase BTC/TRX/ZEC weights; reduce ETH/ADA/LINK/DOGE."
                if btc_dom_val > 0
                else "Spread across high-momentum alts (ADA/XLM/XRP/DOGE/ETH); reduce BTC."
            )
            btc_dom_extra = (
                f"BTC vs Equal-Weight basket (30d): {btc_dom_val:+.1%} → {_dom_label}. {_dom_hint}"
            )
        else:
            btc_dom_extra = ""

        eval1 = await self.agent1.analyze(
            bundle.market_snapshots,
            kline_images=bundle.kline_images,
            lookback_days=self.lookback_days,
            extra_context=btc_dom_extra if btc_dom_extra else None,
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


        w1, cw1 = self._extract_private_weights(eval1, "agent1")
        w2, cw2 = self._extract_private_weights(eval2, "agent2")
        w3, cw3 = self._extract_private_weights(eval3, "agent3")

        logger.info(
            f"[ShapleyCouncil] Stage 1 done | "
            f"conf1={eval1.confidence:.2f} conf2={eval2.confidence:.2f} conf3={eval3.confidence:.2f}"
        )


        current_regime   = getattr(bundle, "market_regime", None) or "volatile"
        regime_score     = getattr(bundle, "regime_score", 0.0)
        transition_boost = getattr(bundle, "regime_transition_boost", None)


        regime_multipliers = MarketRegimeDetector.get_soft_multipliers(
            regime_score, transition_boost,
            bull_mults    = self._regime_bull_mults,
            neutral_mults = self._regime_neutral_mults,
            bear_mults    = self._regime_bear_mults,
        )


        _dom1 = eval1.dominant_dimension or "technical"
        _dom2 = eval2.dominant_dimension or "onchain"
        _dom3 = eval3.dominant_dimension or "macro"


        _base_aw = self.shapley_ledger.get_next_agent_weights(
            regime_multipliers=regime_multipliers, dominance_multipliers=None
        )


        _dom_weighted: Dict[str, float] = {}
        for _d, _w in [(_dom1, _base_aw["agent1"]),
                       (_dom2, _base_aw["agent2"]),
                       (_dom3, _base_aw["agent3"])]:
            _dom_weighted[_d] = _dom_weighted.get(_d, 0.0) + _w


        _max_dim   = max(_dom_weighted, key=_dom_weighted.get)
        _max_share = _dom_weighted[_max_dim]
        _uniform   = 1.0 / max(len(_dom_weighted), 1)
        _dom_strength = max(0.0, min(1.0, (_max_share - _uniform) / (1.0 - _uniform)))

        dominant_dimension = _max_dim if _dom_strength > 0.01 else "other"

        _full_mults = _DOMINANCE_MULTIPLIERS.get(_max_dim, _DOMINANCE_MULTIPLIERS["other"])
        _neutral    = _DOMINANCE_MULTIPLIERS["other"]   # all 1.0
        dominance_multipliers = {
            k: _neutral[k] + _dom_strength * (_full_mults[k] - _neutral[k])
            for k in _full_mults
        }

        _dominance_override = dominance_multipliers if _dom_strength > 0.01 else None

        logger.info(
            f"[ShapleyCouncil] Market regime: {current_regime} (score={regime_score:+.3f}) | "
            f"regime_mults={{{','.join(f'{k}:{v:.2f}' for k,v in regime_multipliers.items())}}} | "
            f"Dominant dim: {dominant_dimension} "
            f"(weighted: {{{', '.join(f'{d}:{s:.2f}' for d,s in sorted(_dom_weighted.items(), key=lambda x:-x[1]))}}})"
            + (f" | dom_mults={{{','.join(f'{k}:{v:.2f}' for k,v in dominance_multipliers.items())}}}"
               if dominant_dimension != "other" else " [no override — <50% threshold]")
        )


        def _norm_regime_vote(s: str) -> str:
            s = (s or "").lower()
            if any(x in s for x in ("bull", "uptrend", "rising", "bullish")):
                return "bull"
            if any(x in s for x in ("bear", "downtrend", "falling", "bearish")):
                return "bear"
            return "volatile"

        _vote1 = _norm_regime_vote(eval1.parsed.get("market_regime", ""))
        _vote2 = _norm_regime_vote(eval2.parsed.get("market_regime", ""))
        _vote3 = _norm_regime_vote(eval3.parsed.get("market_regime", ""))


        _vote_tally: Dict[str, float] = {}
        for _ag, _v in [("agent1", _vote1), ("agent2", _vote2), ("agent3", _vote3)]:
            _vote_tally[_v] = _vote_tally.get(_v, 0.0) + _base_aw[_ag]


        _regime_consensus = max(_vote_tally.values()) if _vote_tally else 1.0


        logger.info("[ShapleyCouncil] Stage 2: Pairwise Socratic debates (1↔2, 1↔3, 2↔3)")
        pair_eval_12 = await self._pairwise_debate(
            eval_a=eval1, eval_b=eval2,
            pair_id="1_2",
            analyst_a_name="Market Technical Analyst",
            analyst_b_name="On-chain & Sentiment Analyst",
            analyst_a_role="market price data, OHLCV, K-line charts",
            analyst_b_role="on-chain flows, blockchain activity, sentiment data",
        )
        pair_eval_13 = await self._pairwise_debate(
            eval_a=eval1, eval_b=eval3,
            pair_id="1_3",
            analyst_a_name="Market Technical Analyst",
            analyst_b_name="Macro Environment Analyst",
            analyst_a_role="market price data, OHLCV, K-line charts",
            analyst_b_role="macro indicators (FRED), VIX, USD index, Fear & Greed",
        )
        pair_eval_23 = await self._pairwise_debate(
            eval_a=eval2, eval_b=eval3,
            pair_id="2_3",
            analyst_a_name="On-chain & Sentiment Analyst",
            analyst_b_name="Macro Environment Analyst",
            analyst_a_role="on-chain flows, blockchain activity, sentiment data",
            analyst_b_role="macro indicators (FRED), VIX, USD index, Fear & Greed",
        )


        _sh_a1 = self.shapley_ledger._compute_sharpe(self.shapley_ledger._s1_ret["agent1"])
        _sh_a2 = self.shapley_ledger._compute_sharpe(self.shapley_ledger._s1_ret["agent2"])
        _sh_a3 = self.shapley_ledger._compute_sharpe(self.shapley_ledger._s1_ret["agent3"])


        w12, cw12 = self._extract_pair_weights(
            pair_eval_12, "1_2", (w1, cw1), (w2, cw2), _sh_a1, _sh_a2)
        w13, cw13 = self._extract_pair_weights(
            pair_eval_13, "1_3", (w1, cw1), (w3, cw3), _sh_a1, _sh_a3)
        w23, cw23 = self._extract_pair_weights(
            pair_eval_23, "2_3", (w2, cw2), (w3, cw3), _sh_a2, _sh_a3)

        logger.info("[ShapleyCouncil] Stage 2 done")


        record = ShapleyStageRecord(
            eval_date = bundle.eval_date,
            prices    = dict(bundle.prices),
            # Stage 1
            stage1_weights  = {"agent1": w1,  "agent2": w2,  "agent3": w3},
            stage1_cash     = {"agent1": cw1, "agent2": cw2, "agent3": cw3},
            stage1_eval_raw = {
                "agent1": eval1.raw_text,
                "agent2": eval2.raw_text,
                "agent3": eval3.raw_text,
            },
            stage1_parsed   = {
                "agent1": eval1.parsed,
                "agent2": eval2.parsed,
                "agent3": eval3.parsed,
            },
            # Stage 2
            stage2_weights  = {"1_2": w12,  "1_3": w13,  "2_3": w23},
            stage2_cash     = {"1_2": cw12, "1_3": cw13, "2_3": cw23},
            stage2_eval_raw = {
                "1_2": pair_eval_12.raw_text,
                "1_3": pair_eval_13.raw_text,
                "2_3": pair_eval_23.raw_text,
            },
            stage2_parsed   = {
                "1_2": pair_eval_12.parsed,
                "1_3": pair_eval_13.parsed,
                "2_3": pair_eval_23.parsed,
            },
        )


        logger.info("[ShapleyCouncil] Stage 3: Agent 4 synthesis with Shapley value context")
        shapley_report = self.shapley_ledger.format_shapley_report(for_agent4_prompt=True)

        agent_weights  = self.shapley_ledger.get_next_agent_weights(
            regime_multipliers    = regime_multipliers,
            dominance_multipliers = _dominance_override,
        )
        pair_weights   = self.shapley_ledger.get_next_pair_weights()
        perf_summary   = self.performance_eval.get_feedback_summary()


        # eff_i = shapley_i * conf_i / Σ(shapley_j * conf_j)

        aw1_raw = agent_weights.get("agent1", 1/3)
        aw2_raw = agent_weights.get("agent2", 1/3)
        aw3_raw = agent_weights.get("agent3", 1/3)


        #   conviction_i = L1(w_i, 1/N) / max_possible_deviation


        _N  = len(ALL_TOKENS)
        _eq = 1.0 / _N

        def _conviction(weights_dict: Dict[str, float], llm_conf: float) -> float:
            """L1 deviation from equal-weight, scaled to [0.3, 1.0]."""
            l1 = sum(abs(weights_dict.get(sym, 0.0) - _eq) for sym in ALL_TOKENS)
            max_l1 = 2.0 * (1.0 - _eq)
            dev_score = l1 / max_l1 if max_l1 > 0 else 0.5

            blended = 0.70 * dev_score + 0.30 * llm_conf
            return max(0.3, min(1.0, blended))

        conf1 = _conviction(w1, float(eval1.parsed.get("stage1_confidence", eval1.confidence) or 0.5))
        conf2 = _conviction(w2, float(eval2.parsed.get("stage1_confidence", eval2.confidence) or 0.5))
        conf3 = _conviction(w3, float(eval3.parsed.get("stage1_confidence", eval3.confidence) or 0.5))

        eff1 = aw1_raw * conf1
        eff2 = aw2_raw * conf2
        eff3 = aw3_raw * conf3
        eff_sum = eff1 + eff2 + eff3
        if eff_sum > 0:
            aw1, aw2, aw3 = eff1 / eff_sum, eff2 / eff_sum, eff3 / eff_sum
        else:
            aw1, aw2, aw3 = 1/3, 1/3, 1/3
        logger.debug(
            f"[ShapleyCouncil] Conf-weighted agent blends: "
            f"a1={aw1:.3f}(sh={aw1_raw:.3f}×c={conf1:.2f}) "
            f"a2={aw2:.3f}(sh={aw2_raw:.3f}×c={conf2:.2f}) "
            f"a3={aw3:.3f}(sh={aw3_raw:.3f}×c={conf3:.2f})"
        )
        s1_ensemble = {
            sym: aw1 * float(w1.get(sym, 0.0))
                 + aw2 * float(w2.get(sym, 0.0))
                 + aw3 * float(w3.get(sym, 0.0))
            for sym in ALL_TOKENS
        }
        s1_cash = aw1 * float(cw1) + aw2 * float(cw2) + aw3 * float(cw3)


        pw12 = pair_weights.get("1_2", 1/3)
        pw13 = pair_weights.get("1_3", 1/3)
        pw23 = pair_weights.get("2_3", 1/3)
        s2_ensemble = {
            sym: pw12 * float(w12.get(sym, 0.0))
                 + pw13 * float(w13.get(sym, 0.0))
                 + pw23 * float(w23.get(sym, 0.0))
            for sym in ALL_TOKENS
        }
        s2_cash = pw12 * float(cw12) + pw13 * float(cw13) + pw23 * float(cw23)


        _wta_result = self.shapley_ledger.get_selective_wta()
        if _wta_result is not None:
            _wta_key, _wta_type = _wta_result
            _WTA_W = 0.80
            if _wta_type == "s1":

                _others_sum = sum(v for k, v in agent_weights.items() if k != _wta_key)
                _new_aw: Dict[str, float] = {}
                for _a in ["agent1", "agent2", "agent3"]:
                    if _a == _wta_key:
                        _new_aw[_a] = _WTA_W
                    else:
                        _new_aw[_a] = (1.0 - _WTA_W) * (agent_weights.get(_a, 1/3) / _others_sum) \
                                       if _others_sum > 1e-9 else (1.0 - _WTA_W) / 2
                _aw_sum = sum(_new_aw.values())
                agent_weights = {k: v / _aw_sum for k, v in _new_aw.items()}
            else:

                _others_sum = sum(v for k, v in pair_weights.items() if k != _wta_key)
                _new_pw: Dict[str, float] = {}
                for _p in ["1_2", "1_3", "2_3"]:
                    if _p == _wta_key:
                        _new_pw[_p] = _WTA_W
                    else:
                        _new_pw[_p] = (1.0 - _WTA_W) * (pair_weights.get(_p, 1/3) / _others_sum) \
                                       if _others_sum > 1e-9 else (1.0 - _WTA_W) / 2
                _pw_sum = sum(_new_pw.values())
                pair_weights = {k: v / _pw_sum for k, v in _new_pw.items()}

        s1_blend, agent4_blend = self._compute_blend_ratios(agent_weights, pair_weights)


        _a4_before_disc = agent4_blend
        agent4_blend = agent4_blend * (0.5 + 0.5 * _regime_consensus)
        logger.info(
            f"[ShapleyCouncil] Regime votes: {_vote1}/{_vote2}/{_vote3} "
            f"(consensus={_regime_consensus:.2f}) | "
            f"a4_blend {_a4_before_disc:.3f}→{agent4_blend:.3f} | "
            f"s1_blend={s1_blend:.3f}"
        )

        shapley_ensemble = {
            sym: s1_blend * s1_ensemble.get(sym, 0.0)
                 + (1.0 - s1_blend) * s2_ensemble.get(sym, 0.0)
            for sym in ALL_TOKENS
        }
        shapley_ensemble_cash = s1_blend * s1_cash + (1.0 - s1_blend) * s2_cash

        _ens_total = sum(shapley_ensemble.values()) + shapley_ensemble_cash
        if _ens_total > 0:
            shapley_ensemble = {k: round(v / _ens_total, 6) for k, v in shapley_ensemble.items()}
            shapley_ensemble_cash = round(shapley_ensemble_cash / _ens_total, 6)

        decision = await self.agent4.analyze_with_shapley(
            agent_evals  = {"agent1": eval1, "agent2": eval2, "agent3": eval3},
            pair_evals   = {"1_2": pair_eval_12, "1_3": pair_eval_13, "2_3": pair_eval_23},
            shapley_report = shapley_report,
            agent_weights  = agent_weights,
            pair_weights   = pair_weights,
            performance_feedback = perf_summary,
            current_regime = current_regime,
            regime_score   = regime_score,

            dominant_dimension = dominant_dimension,
            dominant_votes     = f"agent1={_dom1}, agent2={_dom2}, agent3={_dom3}",
        )


        record.stage3_eval_raw = decision.raw_text
        record.stage3_parsed   = decision.parsed


        if decision.parsed.get("parse_error"):
            logger.warning(
                "[ShapleyCouncil] Stage 3 Agent 4 parse failed. "
                "Using Shapley-weighted Stage 1+2 ensemble as final portfolio."
            )
            stage3_weights = shapley_ensemble
            stage3_cash    = shapley_ensemble_cash
        else:

            _a4_w  = decision.parsed.get("portfolio_weights", {})
            _a4_cw = float(decision.parsed.get("cash_weight", 0.0))
            blend  = agent4_blend
            raw_w  = {
                sym: blend * float(_a4_w.get(sym, 0.0))
                     + (1.0 - blend) * shapley_ensemble.get(sym, 0.0)
                for sym in ALL_TOKENS
            }
            raw_cash = blend * _a4_cw + (1.0 - blend) * shapley_ensemble_cash
            _blend_total = sum(raw_w.values()) + raw_cash
            if _blend_total > 0:
                stage3_weights = {k: round(v / _blend_total, 6) for k, v in raw_w.items()}
                stage3_cash    = round(raw_cash / _blend_total, 6)
            else:
                stage3_weights = shapley_ensemble
                stage3_cash    = shapley_ensemble_cash

        record.stage3_weights = stage3_weights
        record.stage3_cash    = stage3_cash

        self.shapley_ledger.add_record(record)


        shapley_snapshot = self.shapley_ledger.get_shapley_snapshot()
        logger.info(
            # φᵢ = Shapley values (game theory output)
            # ρ  = Sharpe ratios  (performance display metric)
            f"[ShapleyCouncil] φ(Shapley): S1=["
            f"{shapley_snapshot['stage1_shapley']['agent1']:+.3f}, "
            f"{shapley_snapshot['stage1_shapley']['agent2']:+.3f}, "
            f"{shapley_snapshot['stage1_shapley']['agent3']:+.3f}]  "
            f"ρ(Sharpe): S1=["
            f"{shapley_snapshot['stage1_sharpe']['agent1']:+.3f}, "
            f"{shapley_snapshot['stage1_sharpe']['agent2']:+.3f}, "
            f"{shapley_snapshot['stage1_sharpe']['agent3']:+.3f}]  "
            f"S3_ρ={shapley_snapshot['stage3_sharpe']:+.3f}"
        )


        dominator = decision.parsed.get("dominator", self.current_weights)
        self._update_dominator(dominator)

        return {

            "stage1_evals": {
                "agent1": eval1, "agent2": eval2, "agent3": eval3,
            },
            "stage2_decision":  decision,
            "current_dominator": dict(self.current_weights),
            "portfolio_weights": stage3_weights,
            "cash_weight":       stage3_cash,
            "market_regime":     current_regime,
            "risk_level":        decision.parsed.get("risk_level", "unknown"),

            "stage2_pair_evals": {
                "1_2": pair_eval_12, "1_3": pair_eval_13, "2_3": pair_eval_23,
            },
            "stage3_eval":            decision,
            "shapley_snapshot":       shapley_snapshot,
            "stage_record":           record,
            "agent_weights_used":       agent_weights,
            "pair_weights_used":        pair_weights,
            "regime_multipliers_used":  regime_multipliers,
            "dominant_dimension":       dominant_dimension,
            "dominant_dimension_votes": {"agent1": _dom1, "agent2": _dom2, "agent3": _dom3},
            "dominance_override_used":  _dominance_override,
            "shapley_ensemble":         shapley_ensemble,
            "shapley_ensemble_cash":    shapley_ensemble_cash,
        }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def run(self, symbols: List[str] = None) -> Dict[str, Any]:
        """"""
        if self.data_manager is None:
            raise RuntimeError(
                "data_manager is required for run(). "
                "Pass data_manager= to RegimeCouncil(), or use run_with_bundle() for backtest."
            )

        symbols = symbols or (
            self.data_manager.config.tokens if hasattr(self.data_manager, "config") else ALL_TOKENS
        )
        logger.info("=" * 60)
        logger.info("RegimeCouncil — Starting Real-time Analysis")
        logger.info("=" * 60)


        market_data  = await self.data_manager.get_data_for_market_observer(symbols)
        onchain_data = await self.data_manager.get_data_for_onchain_observer(symbols)
        env_data     = await self.data_manager.get_data_for_environment_observer()

        kline_images: Dict[str, Any] = {}
        for sym in symbols:
            img = await self.data_manager.get_kline_image(sym, "1d")
            if img:
                kline_images[sym] = img


        today = datetime.date.today()
        bundle = DateRangeBundle(
            eval_date      = today,
            market_snapshots = market_data,
            kline_images   = kline_images,
            onchain_metrics = onchain_data,
            env_snapshot   = env_data,
            prices         = {
                sym: snap.price for sym, snap in market_data.items()
                if hasattr(snap, "price")
            },
        )
        return await self.run_with_bundle(bundle)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Dict[str, Any] = None,
    ):
        """"""
        self.performance_eval.record(
            date=date, prediction=prediction or {},
            actual_return=actual_return, portfolio_value=portfolio_value,
        )

    def record_trade_result(self, pnl: float, prediction: Dict[str, Any] = None):
        """"""
        self.performance_eval.record(
            date=datetime.date.today(),
            prediction=prediction or {},
            actual_return=pnl / 100_000,
        )

    def get_shapley_report(self, verbose: bool = False) -> str:
        """"""
        return self.shapley_ledger.format_shapley_report(for_agent4_prompt=not verbose)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def _pairwise_debate(
        self,
        eval_a: AgentEvaluation,
        eval_b: AgentEvaluation,
        pair_id: str,
        analyst_a_name: str,
        analyst_b_name: str,
        analyst_a_role: str,
        analyst_b_role: str,
    ) -> AgentEvaluation:
        """"""

        a_priv = eval_a.parsed.get("private_portfolio_weights", {})
        b_priv = eval_b.parsed.get("private_portfolio_weights", {})
        a_cash = eval_a.parsed.get("private_cash_weight", 0.0)
        b_cash = eval_b.parsed.get("private_cash_weight", 0.0)
        a_conf = eval_a.parsed.get("stage1_confidence", eval_a.confidence)
        b_conf = eval_b.parsed.get("stage1_confidence", eval_b.confidence)

        system_prompt = PAIRWISE_DEBATE_SYSTEM.format(
            analyst_a_role=analyst_a_role,
            analyst_b_role=analyst_b_role,
        )
        user_prompt = PAIRWISE_DEBATE_USER.format(
            pair_id               = pair_id,
            analyst_a_name        = analyst_a_name,
            analyst_b_name        = analyst_b_name,
            analyst_a_role        = analyst_a_role,
            analyst_b_role        = analyst_b_role,
            # Use compact eval summaries instead of full raw_text to reduce
            # Stage 2 prompt length (critical for small models).
            analyst_a_eval        = self._compact_eval(eval_a),
            analyst_b_eval        = self._compact_eval(eval_b),
            analyst_a_private_weights = json.dumps(a_priv, ensure_ascii=False),
            analyst_b_private_weights = json.dumps(b_priv, ensure_ascii=False),
            analyst_a_cash_weight = float(a_cash),
            analyst_b_cash_weight = float(b_cash),
            analyst_a_confidence  = float(a_conf),
            analyst_b_confidence  = float(b_conf),
        )

        response = await self._llm.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


        parsed = self.agent4._parse_json(response)
        return AgentEvaluation(
            agent_name       = f"pair_{pair_id}",
            raw_text         = response,
            parsed           = parsed,
            confidence       = float(parsed.get("pair_confidence", 0.0)),
            key_observations = [parsed.get("synthesis", "")],
        )

    @staticmethod
    def _compact_eval(eval_result: AgentEvaluation, max_raw_chars: int = 600) -> str:
        """
        Returns a compact JSON summary of an AgentEvaluation for Stage 2 debate prompts.
        Using parsed fields (not raw_text) dramatically reduces Stage 2 input length,
        which is critical for small models like qwen3-vl:2b.
        Falls back to truncated raw_text if parsing failed.
        """
        p = eval_result.parsed
        if p.get("parse_error") or not p:
            return eval_result.raw_text[:max_raw_chars]

        compact: Dict[str, Any] = {}
        for regime_key in ("market_regime", "onchain_regime", "macro_regime"):
            if p.get(regime_key):
                compact["regime"] = p[regime_key]
                break
        compact["confidence"] = round(
            float(p.get("regime_confidence", p.get("confidence", 0.0))), 3
        )
        obs = p.get("key_observations", [])
        compact["key_observations"] = obs[:3]
        # Include any extra regime-specific top-level signal
        for extra_key in ("btc_macro_signal", "fred_signals", "fear_greed"):
            if extra_key in p:
                compact[extra_key] = p[extra_key]
        return json.dumps(compact, indent=2, ensure_ascii=False)

    @staticmethod
    def _extract_private_weights(
        eval_result: AgentEvaluation,
        agent_id: str,
    ) -> Tuple[Dict[str, float], float]:
        """"""
        parsed = eval_result.parsed
        priv_w = parsed.get("private_portfolio_weights", {})
        priv_c = float(parsed.get("private_cash_weight", 0.0))

        if not priv_w or parsed.get("parse_error"):
            logger.warning(
                f"[ShapleyCouncil] {agent_id} has no private_portfolio_weights "
                f"(parse_error={parsed.get('parse_error', False)}). Using equal weight."
            )
            n = len(ALL_TOKENS)
            priv_w = {sym: round(1.0 / n, 6) for sym in ALL_TOKENS}
            priv_c = 0.0
        else:

            total = sum(float(v) for v in priv_w.values()) + priv_c
            if total > 0:
                priv_w = {k: round(float(v) / total, 6) for k, v in priv_w.items()}
                priv_c = round(priv_c / total, 6)

        return priv_w, priv_c

    @staticmethod
    def _extract_pair_weights(
        pair_eval: AgentEvaluation,
        pair_id: str,
        agent_a_weights: Tuple[Dict[str, float], float] = None,
        agent_b_weights: Tuple[Dict[str, float], float] = None,
        agent_a_sharpe: float = 0.0,
        agent_b_sharpe: float = 0.0,
    ) -> Tuple[Dict[str, float], float]:
        """"""
        parsed = pair_eval.parsed
        jw     = parsed.get("joint_portfolio_weights", {})
        jc     = float(parsed.get("joint_cash_weight", 0.0))

        if not jw or parsed.get("parse_error"):
            if agent_a_weights and agent_b_weights:
                wa, ca = agent_a_weights
                wb, cb = agent_b_weights

                sa = max(0.0, agent_a_sharpe)
                sb = max(0.0, agent_b_sharpe)
                st = sa + sb
                wa_w = sa / st if st > 1e-9 else 0.5
                wb_w = sb / st if st > 1e-9 else 0.5
                jw = {
                    sym: wa_w * float(wa.get(sym, 0.0)) + wb_w * float(wb.get(sym, 0.0))
                    for sym in ALL_TOKENS
                }
                jc = wa_w * float(ca) + wb_w * float(cb)
                total = sum(jw.values()) + jc
                if total > 0:
                    jw = {k: round(v / total, 6) for k, v in jw.items()}
                    jc = round(jc / total, 6)
                logger.warning(
                    f"[ShapleyCouncil] Pair {pair_id} has no joint weights. "
                    f"Using Sharpe-weighted fallback (wa={wa_w:.2f}, wb={wb_w:.2f}, "
                    f"sharpe_a={agent_a_sharpe:.2f}, sharpe_b={agent_b_sharpe:.2f})."
                )
            else:
                logger.warning(
                    f"[ShapleyCouncil] Pair {pair_id} has no joint weights. Using equal weight."
                )
                n = len(ALL_TOKENS)
                jw = {sym: round(1.0 / n, 6) for sym in ALL_TOKENS}
                jc = 0.0
        else:
            total = sum(float(v) for v in jw.values()) + jc
            if total > 0:
                jw = {k: round(float(v) / total, 6) for k, v in jw.items()}
                jc = round(jc / total, 6)

        return jw, jc

    def _update_dominator(self, new_weights: Dict[str, float]):
        """"""
        alpha = 0.3
        for key in self.current_weights:
            if key in new_weights:
                old = self.current_weights[key]
                new = float(new_weights[key])
                self.current_weights[key] = round(old * (1 - alpha) + new * alpha, 4)
        total = sum(self.current_weights.values())
        if total > 0:
            for key in self.current_weights:
                self.current_weights[key] = round(self.current_weights[key] / total, 4)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
ShapleyCouncil = RegimeCouncil
SharpeCouncil  = RegimeCouncil
