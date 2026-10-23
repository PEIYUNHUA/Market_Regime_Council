""""""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentEvaluation
from ..llm.base import BaseLLM
from ..prompts.templates import (
    INVESTMENT_ANALYST_SYSTEM, INVESTMENT_ANALYST_USER,
    INVESTMENT_ANALYST_SHAPLEY_USER,
)
from ..config import ALL_TOKENS


class InvestmentAnalyst(BaseAgent):

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = None,
        allow_short: bool = False,
        max_short: float = 0.20,
    ):
        from ..prompts.templates import INVESTMENT_ANALYST_SHORT_SYSTEM
        if system_prompt is None:
            system_prompt = INVESTMENT_ANALYST_SHORT_SYSTEM if allow_short else INVESTMENT_ANALYST_SYSTEM
        super().__init__(
            name="investment_analyst (agent 4)", llm=llm,
            system_prompt=system_prompt,
        )
        self._allow_short = allow_short
        self._max_short = max_short
        self._performance_log: List[Dict[str, Any]] = []

    async def analyze(
        self,
        agent_evals: Dict[str, AgentEvaluation],
        performance_feedback: str = "No performance history yet.",
        portfolio_data: str = "No active portfolio.",
    ) -> AgentEvaluation:
        """"""
        _empty = AgentEvaluation("N/A", "N/A", {})
        user_prompt = INVESTMENT_ANALYST_USER.format(
            agent1_eval=agent_evals.get("agent1", _empty).raw_text,
            agent2_eval=agent_evals.get("agent2", _empty).raw_text,
            agent3_eval=agent_evals.get("agent3", _empty).raw_text,
            performance_feedback=performance_feedback,
        )
        response = await self.llm.chat_text(system_prompt=self.system_prompt, user_prompt=user_prompt)

        parsed = self._parse_json(response)


        portfolio_weights = self._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )
        parsed["portfolio_weights"] = portfolio_weights["token_weights"]
        parsed["cash_weight"] = portfolio_weights["cash_weight"]

        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("dominator", {}),
            confidence=parsed.get("regime_confidence", parsed.get("confidence", 0.0)),
            key_observations=[
                f"Market regime: {parsed.get('market_regime', 'unknown')}",
                f"Risk level: {parsed.get('risk_level', 'unknown')}",
                f"Cash weight: {parsed.get('cash_weight', 0):.1%}",
                parsed.get("rebalance_rationale", ""),
            ],
        )
        self._history.append(eval_result)
        return eval_result

    def _normalize_weights(self, raw_weights: Dict[str, float],
                           cash_weight: float) -> Dict[str, Any]:
        """"""
        MAX_TOKEN = 0.30
        MAX_CASH = 0.08
        MAX_SHORT = self._max_short if self._allow_short else 0.0
        min_token = -MAX_SHORT


        token_weights = {sym: float(raw_weights.get(sym, 0.0)) for sym in ALL_TOKENS}


        token_weights = {k: max(min_token, min(MAX_TOKEN, v)) for k, v in token_weights.items()}
        cash = max(0.0, min(MAX_CASH, float(cash_weight) if cash_weight else 0.0))

        target_net = 1.0 - cash
        net = sum(token_weights.values())

        if abs(net) > 1e-9:

            scale = target_net / net
            token_weights = {k: round(v * scale, 6) for k, v in token_weights.items()}
        else:

            per = target_net / len(ALL_TOKENS)
            token_weights = {sym: round(per, 6) for sym in ALL_TOKENS}


        for _iter in range(20):
            token_weights = {k: max(min_token, min(MAX_TOKEN, v)) for k, v in token_weights.items()}
            deficit = target_net - sum(token_weights.values())
            if abs(deficit) < 1e-9:
                break

            room = {}
            for k, v in token_weights.items():
                if deficit > 0 and v < MAX_TOKEN - 1e-9:
                    room[k] = MAX_TOKEN - v
                elif deficit < 0 and v > min_token + 1e-9:
                    room[k] = v - min_token
            total_room = sum(room.values())
            if total_room < 1e-9:
                break
            for k, r in room.items():
                token_weights[k] = round(
                    token_weights[k] + deficit * (r / total_room), 6
                )


        residual = round(target_net - sum(token_weights.values()), 8)
        if abs(residual) > 1e-9:

            def _room(v: float) -> float:
                return min(MAX_TOKEN - v, v - min_token)
            anchor = max(token_weights, key=lambda k: _room(token_weights[k]))
            new_v = round(token_weights[anchor] + residual, 8)
            token_weights[anchor] = max(min_token, min(MAX_TOKEN, new_v))

        return {"token_weights": token_weights, "cash_weight": round(cash, 6)}

    @staticmethod
    def _compact_stage1(e: "AgentEvaluation", max_raw: int = 600) -> str:
        """
        Compact text representation of a Stage 1 agent eval for Agent 4's input.
        Uses parsed fields when available; falls back to truncated raw_text.
        This prevents Agent 4's context window from overflowing when a Stage 1
        agent's raw_text contains a long thinking chain instead of clean JSON.
        """
        p = e.parsed
        if not p or p.get("parse_error"):
            return e.raw_text[:max_raw]
        lines = []
        for regime_key in ("market_regime", "onchain_regime", "macro_regime"):
            if p.get(regime_key):
                lines.append(f"regime={p[regime_key]}, conf={p.get('regime_confidence', 0):.2f}")
                break
        obs = p.get("key_observations", [])
        if obs:
            lines.append("observations: " + " | ".join(str(o) for o in obs[:3]))
        pw = p.get("private_portfolio_weights", {})
        if pw:
            top5 = sorted(pw.items(), key=lambda x: -float(x[1]))[:5]
            lines.append("top weights: " + ", ".join(f"{k}={float(v):.3f}" for k, v in top5))
            lines.append(f"cash={p.get('private_cash_weight', 0):.3f}")
        return "\n".join(lines) if lines else e.raw_text[:max_raw]

    @staticmethod
    def _compact_pair(e: "AgentEvaluation", max_raw: int = 400) -> str:
        """
        Compact text representation of a Stage 2 pair eval for Agent 4's input.
        """
        p = e.parsed
        if not p or p.get("parse_error"):
            return e.raw_text[:max_raw]
        lines = []
        regime = p.get("pair_regime", "")
        conf   = p.get("pair_confidence", 0)
        if regime:
            lines.append(f"regime={regime}, conf={conf:.2f}")
        synth = p.get("synthesis", "")
        if synth:
            lines.append(f"synthesis: {synth}")
        jw = p.get("joint_portfolio_weights", {})
        if jw:
            top5 = sorted(jw.items(), key=lambda x: -float(x[1]))[:5]
            lines.append("top weights: " + ", ".join(f"{k}={float(v):.3f}" for k, v in top5))
            lines.append(f"cash={p.get('joint_cash_weight', 0):.3f}")
        return "\n".join(lines) if lines else e.raw_text[:max_raw]

    async def analyze_with_shapley(
        self,
        agent_evals:          Dict[str, AgentEvaluation],
        pair_evals:           Dict[str, AgentEvaluation],
        shapley_report:       str,
        agent_weights:        Dict[str, float],
        pair_weights:         Dict[str, float],
        performance_feedback: str = "No performance history yet.",
        current_regime:       str = "volatile",
        regime_score:         float = 0.0,
        dominant_dimension:   str = "",
        dominant_votes:       str = "",
    ) -> AgentEvaluation:
        """"""

        _agent_labels = {
            "agent1": "Alex Chen (Technical/Market)",
            "agent2": "Blake Kim (On-chain/Sentiment)",
            "agent3": "Clara Wang (Macro/Environment)",
        }
        _dom_key = max(agent_weights, key=lambda k: agent_weights.get(k, 0))
        _regime_dominator = f"{_agent_labels.get(_dom_key, _dom_key)} [weight={agent_weights.get(_dom_key, 0):.3f}]"


        _dom_label = dominant_dimension or "unspecified"
        _dom_votes_label = dominant_votes or "N/A"

        _empty = AgentEvaluation("N/A", "N/A", {})
        # Use compact representations to prevent context overflow.
        # raw_text may be a 4000-token thinking chain if JSON parsing failed upstream.
        user_prompt = INVESTMENT_ANALYST_SHAPLEY_USER.format(
            current_regime     = current_regime,
            regime_score       = regime_score,
            regime_dominator   = _regime_dominator,
            dominant_dimension = _dom_label,
            dominant_votes     = _dom_votes_label,
            shapley_report     = shapley_report,
            w_agent1           = agent_weights.get("agent1", 1/3),
            w_agent2           = agent_weights.get("agent2", 1/3),
            w_agent3           = agent_weights.get("agent3", 1/3),
            agent1_eval        = self._compact_stage1(agent_evals.get("agent1", _empty)),
            agent2_eval        = self._compact_stage1(agent_evals.get("agent2", _empty)),
            agent3_eval        = self._compact_stage1(agent_evals.get("agent3", _empty)),
            w_pair_12          = pair_weights.get("1_2", 1/3),
            w_pair_13          = pair_weights.get("1_3", 1/3),
            w_pair_23          = pair_weights.get("2_3", 1/3),
            pair_12_eval       = self._compact_pair(pair_evals.get("1_2", _empty)),
            pair_13_eval       = self._compact_pair(pair_evals.get("1_3", _empty)),
            pair_23_eval       = self._compact_pair(pair_evals.get("2_3", _empty)),
            performance_feedback = performance_feedback,
        )
        response = await self.llm.chat_text(
            system_prompt=self.system_prompt, user_prompt=user_prompt
        )
        parsed = self._parse_json(response)

        portfolio_weights = self._normalize_weights(
            parsed.get("portfolio_weights", {}),
            parsed.get("cash_weight", 0.0),
        )
        parsed["portfolio_weights"] = portfolio_weights["token_weights"]
        parsed["cash_weight"]       = portfolio_weights["cash_weight"]

        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("dominator", {}),
            confidence=parsed.get("regime_confidence", parsed.get("confidence", 0.0)),
            key_observations=[
                f"Market regime: {parsed.get('market_regime', 'unknown')}",
                f"Risk level: {parsed.get('risk_level', 'unknown')}",
                f"Cash weight: {parsed.get('cash_weight', 0):.1%}",
                parsed.get("rebalance_rationale", ""),
            ],
        )
        self._history.append(eval_result)
        return eval_result

    def get_portfolio_weights(self) -> Optional[Dict[str, float]]:
        """"""
        if not self._history:
            return None
        parsed = self._history[-1].parsed
        weights = dict(parsed.get("portfolio_weights", {}))
        weights["CASH"] = parsed.get("cash_weight", 0.0)
        return weights

    def log_performance(self, result: Dict[str, Any]):
        self._performance_log.append(result)

    def get_performance_summary(self) -> str:
        if not self._performance_log:
            return "No performance history yet."
        total = len(self._performance_log)
        profitable = sum(1 for r in self._performance_log if r.get("pnl", 0) > 0)
        total_pnl = sum(r.get("pnl", 0) for r in self._performance_log)
        return (
            f"Total trades: {total} | Win rate: {profitable/total*100:.1f}% | "
            f"Total PnL: ${total_pnl:+,.2f} | Last 5: {json.dumps(self._performance_log[-5:], default=str)}"
        )
