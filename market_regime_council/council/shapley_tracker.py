""""""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def compute_true_shapley_values(
    v1:   float,
    v2:   float,
    v3:   float,
    v12:  float,
    v13:  float,
    v23:  float,
    v123: float,
) -> Tuple[float, float, float]:
    """"""
    phi1 = (1/3)*v1 + (1/6)*(v12-v2) + (1/6)*(v13-v3) + (1/3)*(v123-v23)
    phi2 = (1/3)*v2 + (1/6)*(v12-v1) + (1/6)*(v23-v3) + (1/3)*(v123-v13)
    phi3 = (1/3)*v3 + (1/6)*(v13-v1) + (1/6)*(v23-v2) + (1/3)*(v123-v12)
    return phi1, phi2, phi3

STAGE1_AGENTS = ["agent1", "agent2", "agent3"]
STAGE2_PAIRS  = ["1_2", "1_3", "2_3"]
PAIR_LABELS   = {"1_2": "Market+Onchain", "1_3": "Market+Macro", "2_3": "Onchain+Macro"}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class ShapleyStageRecord:
    """"""
    eval_date: datetime.date
    prices:    Dict[str, float]


    stage1_weights:    Dict[str, Dict[str, float]] = field(default_factory=dict)
    # agent_id -> {token: weight}
    stage1_cash:       Dict[str, float]            = field(default_factory=dict)
    # agent_id -> cash_weight
    stage1_eval_raw:   Dict[str, str]              = field(default_factory=dict)
    # agent_id -> raw LLM text
    stage1_parsed:     Dict[str, Dict[str, Any]]   = field(default_factory=dict)
    # agent_id -> parsed JSON dict


    stage2_weights:    Dict[str, Dict[str, float]] = field(default_factory=dict)
    # pair_id -> {token: weight}
    stage2_cash:       Dict[str, float]            = field(default_factory=dict)
    stage2_eval_raw:   Dict[str, str]              = field(default_factory=dict)
    stage2_parsed:     Dict[str, Dict[str, Any]]   = field(default_factory=dict)


    stage3_weights:    Dict[str, float]            = field(default_factory=dict)
    stage3_cash:       float                       = 0.0
    stage3_eval_raw:   str                         = ""
    stage3_parsed:     Dict[str, Any]              = field(default_factory=dict)


    stage1_returns:    Dict[str, float]            = field(default_factory=dict)
    # agent_id -> period return
    stage2_returns:    Dict[str, float]            = field(default_factory=dict)
    # pair_id  -> period return
    stage3_return:     float                       = 0.0

    def summary(self) -> str:
        """"""
        lines = [f"=== ShapleyStageRecord [{self.eval_date}] ==="]
        lines.append("Stage 1 private weights (single-agent coalitions):")
        for a in STAGE1_AGENTS:
            w = self.stage1_weights.get(a, {})
            cw = self.stage1_cash.get(a, 0.0)
            r  = self.stage1_returns.get(a, float("nan"))
            top = sorted(w.items(), key=lambda x: -x[1])[:3]
            lines.append(f"  {a}: cash={cw:.2%} top3={top}  period_return={r:+.4f}")
        lines.append("Stage 2 pairwise weights (two-agent coalitions):")
        for p in STAGE2_PAIRS:
            w  = self.stage2_weights.get(p, {})
            cw = self.stage2_cash.get(p, 0.0)
            r  = self.stage2_returns.get(p, float("nan"))
            top = sorted(w.items(), key=lambda x: -x[1])[:3]
            lines.append(f"  {PAIR_LABELS[p]}: cash={cw:.2%} top3={top}  return={r:+.4f}")
        lines.append(f"Stage 3 grand coalition: cash={self.stage3_cash:.2%}  return={self.stage3_return:+.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class ShapleyValueLedger:
    """"""

    def __init__(
        self,
        risk_free_rate:          float = 0.0,
        freq:                    int   = 365,
        softmax_temperature:     float = 0.7,
        min_weight:              float = 0.05,
        max_weight:              float = 0.80,
        min_periods:             int   = 2,
        sharpe_weight:           float = 0.4,
        return_weight:           float = 0.6,
        pair_quality_threshold:  float = 0.40,
        bayesian_lambda:         float = 30.0,


        rolling_window:          Optional[int] = None,
        ew_halflife:             Optional[int] = 252,

        #   w_t ∝ exp(-1/hl)^(T-t);  λ = exp(-1/hl)


        agent_prior: Optional[Dict[str, float]] = None,  # deprecated, ignored
        pair_prior:  Optional[Dict[str, float]] = None,  # deprecated, ignored
    ):
        self.risk_free_rate          = risk_free_rate
        self.freq                    = freq
        self.softmax_temperature     = softmax_temperature   # kept for compat
        self.min_weight              = min_weight            # kept for compat
        self.max_weight              = max_weight            # kept for compat
        self.min_periods             = min_periods
        self.sharpe_weight           = sharpe_weight
        self.return_weight           = return_weight
        self.pair_quality_threshold  = pair_quality_threshold  # kept for compat
        self.bayesian_lambda         = bayesian_lambda
        self.rolling_window          = rolling_window
        self.ew_halflife             = ew_halflife


        N_a = len(STAGE1_AGENTS)
        N_p = len(STAGE2_PAIRS)
        self._agent_prior = {a: 1.0 / N_a for a in STAGE1_AGENTS}
        self._pair_prior  = {p: 1.0 / N_p for p in STAGE2_PAIRS}

        self.records: List[ShapleyStageRecord] = []


        self._s1_ret: Dict[str, List[float]] = {a: [] for a in STAGE1_AGENTS}
        self._s2_ret: Dict[str, List[float]] = {p: [] for p in STAGE2_PAIRS}
        self._s3_ret: List[float] = []

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def add_record(self, record: ShapleyStageRecord) -> None:
        """"""
        if self.records:
            self._fill_prev_returns(self.records[-1], record.prices)
        self.records.append(record)
        logger.info(
            f"[ShapleyTracker] Period {record.eval_date} added "
            f"(total={len(self.records)}, return_periods={len(self._s3_ret)})"
        )

    def finalize(self, final_prices: Dict[str, float]) -> None:
        """"""
        if self.records:
            self._fill_prev_returns(self.records[-1], final_prices)
            logger.info("[ShapleyTracker] Finalized last period returns.")

    def get_shapley_snapshot(self) -> Dict[str, Any]:
        """"""

        rho1   = self._compute_sharpe(self._s1_ret["agent1"])
        rho2   = self._compute_sharpe(self._s1_ret["agent2"])
        rho3   = self._compute_sharpe(self._s1_ret["agent3"])
        rho12  = self._compute_sharpe(self._s2_ret["1_2"])
        rho13  = self._compute_sharpe(self._s2_ret["1_3"])
        rho23  = self._compute_sharpe(self._s2_ret["2_3"])
        rho123 = self._compute_sharpe(self._s3_ret)


        v1   = self._coalition_value(self._s1_ret["agent1"])
        v2   = self._coalition_value(self._s1_ret["agent2"])
        v3   = self._coalition_value(self._s1_ret["agent3"])
        v12  = self._coalition_value(self._s2_ret["1_2"])
        v13  = self._coalition_value(self._s2_ret["1_3"])
        v23  = self._coalition_value(self._s2_ret["2_3"])
        v123 = self._coalition_value(self._s3_ret)


        if self._has_enough_history():
            phi1, phi2, phi3 = compute_true_shapley_values(v1, v2, v3, v12, v13, v23, v123)
        else:
            phi1 = phi2 = phi3 = 0.0

        snap = {
            "n_periods":         len(self.records),
            "n_return_periods":  len(self._s3_ret),


            "stage1_shapley":          {"agent1": phi1, "agent2": phi2, "agent3": phi3},
            "stage1_sharpe":           {"agent1": rho1, "agent2": rho2, "agent3": rho3},
            "stage1_coalition_value":  {"agent1": v1,   "agent2": v2,   "agent3": v3},


            "stage2_sharpe":           {"1_2": rho12, "1_3": rho13, "2_3": rho23},
            "stage2_coalition_value":  {"1_2": v12,   "1_3": v13,   "2_3": v23},


            "stage3_sharpe":           rho123,
            "stage3_coalition_value":  v123,


            "stage1_returns":  {a: list(self._s1_ret[a]) for a in STAGE1_AGENTS},
            "stage2_returns":  {p: list(self._s2_ret[p]) for p in STAGE2_PAIRS},
            "stage3_returns":  list(self._s3_ret),
        }


        snap["stage1"]                   = snap["stage1_shapley"]
        snap["stage1_individual_sharpe"] = snap["stage1_sharpe"]
        snap["stage2"]                   = snap["stage2_coalition_value"]
        snap["stage3"]                   = snap["stage3_coalition_value"]

        return snap

    def get_next_agent_weights(
        self,
        regime_multipliers:    Optional[Dict[str, float]] = None,
        dominance_multipliers: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """"""

        base = dict(self._agent_prior)
        N    = len(STAGE1_AGENTS)
        if not self._has_enough_history():
            w = self._apply_regime_multipliers(base, regime_multipliers)
            return self._apply_regime_multipliers(w, dominance_multipliers)


        v1   = self._coalition_value(self._s1_ret["agent1"])
        v2   = self._coalition_value(self._s1_ret["agent2"])
        v3   = self._coalition_value(self._s1_ret["agent3"])
        v12  = self._coalition_value(self._s2_ret["1_2"])
        v13  = self._coalition_value(self._s2_ret["1_3"])
        v23  = self._coalition_value(self._s2_ret["2_3"])
        v123 = self._coalition_value(self._s3_ret)


        phi1, phi2, phi3 = compute_true_shapley_values(v1, v2, v3, v12, v13, v23, v123)


        n     = len(self._s3_ret)
        alpha = 1.0 - math.exp(-n / self.bayesian_lambda)


        phi_pos  = {"agent1": max(0.0, phi1), "agent2": max(0.0, phi2), "agent3": max(0.0, phi3)}
        total_pos = sum(phi_pos.values())
        if total_pos > 1e-10:
            shapley_part = {a: v / total_pos for a, v in phi_pos.items()}
        else:
            shapley_part = base


        weights = {a: alpha * shapley_part[a] + (1.0 - alpha) * self._agent_prior[a]
                   for a in STAGE1_AGENTS}

        _phi_map = {"agent1": phi1, "agent2": phi2, "agent3": phi3}
        logger.debug(
            f"[ShapleyLedger] Bayesian agent weights (n={n}, α={alpha:.3f}): "
            + " ".join(f"{a}={weights[a]:.3f}(φ={_phi_map[a]:+.3f})" for a in STAGE1_AGENTS)
        )

        weights = self._apply_regime_multipliers(weights, regime_multipliers)

        weights = self._apply_regime_multipliers(weights, dominance_multipliers)
        return weights

    def get_next_pair_weights(self) -> Dict[str, float]:
        """"""

        base = dict(self._pair_prior)
        N    = len(STAGE2_PAIRS)
        if not self._has_enough_history():
            return base


        raw = {p: max(0.0, self._coalition_value(self._s2_ret[p])) for p in STAGE2_PAIRS}
        total = sum(raw.values())

        n     = len(self._s3_ret)
        alpha = 1.0 - math.exp(-n / self.bayesian_lambda)

        if total > 1e-10:
            prop = {p: v / total for p, v in raw.items()}
        else:
            prop = base


        return {p: alpha * prop[p] + (1.0 - alpha) * self._pair_prior[p] for p in STAGE2_PAIRS}

    def get_selective_wta(
        self,
        window: int = 30,
        dominance_ratio: float = 1.8,
        wta_weight: float = 0.80,
    ) -> Optional[Tuple[str, str]]:
        """"""

        if len(self._s3_ret) < window + 2:
            return None

        def _rolling_sharpe(rets: List[float]) -> float:
            if len(rets) < window:
                return 0.0
            r = rets[-window:]
            mu = sum(r) / len(r)
            var = sum((x - mu) ** 2 for x in r) / max(1, len(r) - 1)
            sig = math.sqrt(var) if var > 0 else 0.0
            return mu / sig * math.sqrt(self.freq) if sig > 0 else 0.0


        sharpes: Dict[str, float] = {}
        for a in STAGE1_AGENTS:
            sharpes[a] = _rolling_sharpe(self._s1_ret.get(a, []))
        for p in STAGE2_PAIRS:
            sharpes[p] = _rolling_sharpe(self._s2_ret.get(p, []))


        dominant = max(sharpes, key=lambda k: sharpes[k])
        dom_sharpe = sharpes[dominant]
        others_mean = (sum(sharpes.values()) - dom_sharpe) / max(1, len(sharpes) - 1)

        if others_mean <= 0 or dom_sharpe / others_mean < dominance_ratio:
            return None

        stage_type = "s1" if dominant in STAGE1_AGENTS else "s2"
        logger.info(
            "[SelectiveWTA] Triggered: %s 30d_Sharpe=%.3f > %.1f× others_mean=%.3f → soft WTA %.0f%%",
            dominant, dom_sharpe, dominance_ratio, others_mean, wta_weight * 100,
        )
        return (dominant, stage_type)

    def format_shapley_report(self, for_agent4_prompt: bool = True) -> str:
        """"""
        snap = self.get_shapley_snapshot()
        n    = snap["n_return_periods"]

        if n < self.min_periods:
            hist_note = f"Insufficient history ({n} completed period(s), need ≥{self.min_periods}). Using equal weights."
        else:
            hist_note = f"Based on {n} completed period(s)."

        aw = self.get_next_agent_weights()
        pw = self.get_next_pair_weights()

        if for_agent4_prompt:
            lines = [
                "=== Cooperative Game Theory: Three-Stage Shapley Analysis ===",
                hist_note,
                "",
                "KEY DISTINCTION:",
                "  φᵢ (Shapley value)  = Agent i's marginal contribution across ALL 7 coalition permutations",
                "  ρᵢ (Sharpe ratio)   = Agent i's stand-alone risk-adjusted return (display only)",
                "  v(S) (coalition value) = composite input to Shapley formula; v(S) ≠ ρ(S)",
                "",
                "Stage 1 — Shapley Values φᵢ (cooperative game theory, all 7 coalitions):",
                f"  agent1 (Market Tech): φ₁={snap['stage1_shapley']['agent1']:+.4f}  ρ₁={snap['stage1_sharpe']['agent1']:+.3f}  → next_weight={aw['agent1']:.3f}",
                f"  agent2 (On-chain):    φ₂={snap['stage1_shapley']['agent2']:+.4f}  ρ₂={snap['stage1_sharpe']['agent2']:+.3f}  → next_weight={aw['agent2']:.3f}",
                f"  agent3 (Macro):       φ₃={snap['stage1_shapley']['agent3']:+.4f}  ρ₃={snap['stage1_sharpe']['agent3']:+.3f}  → next_weight={aw['agent3']:.3f}",
                "",
                "Stage 2 — Pairwise Coalitions (pair_weight is derived from v(S), NOT from ρ):",
                f"  Market+Onchain (1_2): ρ={snap['stage2_sharpe']['1_2']:+.4f}  v={snap['stage2_coalition_value']['1_2']:+.4f}  → pair_weight={pw['1_2']:.3f}",
                f"  Market+Macro   (1_3): ρ={snap['stage2_sharpe']['1_3']:+.4f}  v={snap['stage2_coalition_value']['1_3']:+.4f}  → pair_weight={pw['1_3']:.3f}",
                f"  Onchain+Macro  (2_3): ρ={snap['stage2_sharpe']['2_3']:+.4f}  v={snap['stage2_coalition_value']['2_3']:+.4f}  → pair_weight={pw['2_3']:.3f}",
                "",
                f"Stage 3 — Grand Coalition Sharpe: ρ₁₂₃={snap['stage3_sharpe']:+.4f}",
                "",
                "INSTRUCTION: Agent weights are derived from Shapley values φᵢ (NOT from Sharpe ratios).",
                "φᵢ measures each agent's marginal contribution across ALL coalition permutations.",
                "Higher φᵢ → higher next_weight → your allocation should reflect this signal strength.",
            ]
        else:
            lines = [
                "=" * 60,
                "  Three-Stage Shapley Value Report (Cooperative Game Theory)",
                "=" * 60,
                f"  Periods with returns: {n}",
                "  ─────────────────────────────────────────────────────────",
                "  Notation:  φ = Shapley value (game theory output)",
                "             ρ = Sharpe ratio  (display metric only)",
                "             v = coalition characteristic function (Shapley input, v ≠ ρ)",
                "  ─────────────────────────────────────────────────────────",
                "",
                "  Stage 1 — Individual Coalitions:",
            ]
            for a in STAGE1_AGENTS:
                rets  = self._s1_ret[a]
                phi_a = snap["stage1_shapley"][a]
                rho_a = snap["stage1_sharpe"][a]
                v_a   = snap["stage1_coalition_value"][a]
                lines.append(
                    f"    {a}: φ={phi_a:+.4f}  ρ={rho_a:+.4f}  v={v_a:+.4f}  "
                    f"nRet={len(rets)}  mean={_mean(rets):+.4f}  std={_std(rets):.4f}"
                )
            lines += ["", "  Stage 2 — Pairwise Coalitions:"]
            for p in STAGE2_PAIRS:
                rets  = self._s2_ret[p]
                rho_p = snap["stage2_sharpe"][p]
                v_p   = snap["stage2_coalition_value"][p]
                lines.append(
                    f"    {PAIR_LABELS[p]}: ρ={rho_p:+.4f}  v={v_p:+.4f}  "
                    f"nRet={len(rets)}  mean={_mean(rets):+.4f}  std={_std(rets):.4f}"
                )
            lines += [
                "",
                f"  Stage 3 — Grand Coalition: ρ={snap['stage3_sharpe']:+.4f}  v={snap['stage3_coalition_value']:+.4f}  "
                f"nRet={len(self._s3_ret)}  mean={_mean(self._s3_ret):+.4f}  std={_std(self._s3_ret):.4f}",
                "",
                f"  Next-period agent weights (from Shapley φ):    {aw}",
                f"  Next-period pair weights  (from coalition v):   {pw}",
                "=" * 60,
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _fill_prev_returns(
        self,
        prev: ShapleyStageRecord,
        next_prices: Dict[str, float],
    ) -> None:
        """"""

        def _port_return(
            token_weights: Dict[str, float],
            cash_w: float,
        ) -> float:
            r = 0.0
            for sym, w in token_weights.items():
                if w == 0.0:
                    continue
                p0 = prev.prices.get(sym, 0.0)
                p1 = next_prices.get(sym, 0.0)
                if p0 > 0 and p1 > 0:

                    r += w * (p1 / p0 - 1.0)

            return r


        for a in STAGE1_AGENTS:
            w  = prev.stage1_weights.get(a, {})
            cw = prev.stage1_cash.get(a, 0.0)
            r  = _port_return(w, cw)
            self._s1_ret[a].append(r)
            prev.stage1_returns[a] = r


        for p in STAGE2_PAIRS:
            w  = prev.stage2_weights.get(p, {})
            cw = prev.stage2_cash.get(p, 0.0)
            r  = _port_return(w, cw)
            self._s2_ret[p].append(r)
            prev.stage2_returns[p] = r


        r3 = _port_return(prev.stage3_weights, prev.stage3_cash)
        self._s3_ret.append(r3)
        prev.stage3_return = r3

    def _has_enough_history(self) -> bool:
        return all(len(v) >= self.min_periods for v in self._s1_ret.values())

    def _compute_sharpe(self, returns: List[float]) -> float:
        """"""
        if self.rolling_window is not None and len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
        if len(returns) < 2:
            return 0.0
        rf_period = (1.0 + self.risk_free_rate) ** (1.0 / self.freq) - 1.0
        excess = [r - rf_period for r in returns]
        mu    = sum(excess) / len(excess)
        var   = sum((e - mu) ** 2 for e in excess) / (len(excess) - 1)
        sigma = math.sqrt(var) if var > 0 else 0.0
        if sigma < 1e-9:
            return 0.0
        return max(-50.0, min(50.0, mu / sigma * math.sqrt(self.freq)))

    def _coalition_value(self, returns: List[float]) -> float:
        """"""
        if self.rolling_window is not None and len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
        if len(returns) < 2:
            return 0.0

        rf_period = (1.0 + self.risk_free_rate) ** (1.0 / self.freq) - 1.0
        excess = [r - rf_period for r in returns]

        if self.ew_halflife is not None:

            n     = len(excess)
            decay = math.exp(-1.0 / self.ew_halflife)

            ws  = [decay ** (n - 1 - i) for i in range(n)]
            sw  = sum(ws)
            mu  = sum(w * e for w, e in zip(ws, excess)) / sw

            var = sum(w * (e - mu) ** 2 for w, e in zip(ws, excess)) / sw
        else:

            mu  = sum(excess) / len(excess)
            var = sum((e - mu) ** 2 for e in excess) / (len(excess) - 1)

        sigma     = math.sqrt(var) if var > 0 else 0.0
        ann_sigma = sigma * math.sqrt(self.freq)


        if ann_sigma < 1e-3:
            sharpe = 0.0
        else:
            sharpe = max(-50.0, min(50.0, mu / sigma * math.sqrt(self.freq)))


        ann_ret_norm = max(-25.0, min(25.0, mu * self.freq))

        return self.sharpe_weight * sharpe + self.return_weight * ann_ret_norm

    def _normalize_by_shapley_value(self, shapley_vals: Dict[str, float]) -> Dict[str, float]:
        """"""
        weights = _softmax(shapley_vals, temperature=self.softmax_temperature)
        weights = {k: max(self.min_weight, min(self.max_weight, v)) for k, v in weights.items()}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _apply_regime_multipliers(
        self,
        weights: Dict[str, float],
        regime_multipliers: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """"""
        if not regime_multipliers:
            return weights
        adjusted = {
            k: max(0.0, v * regime_multipliers.get(k, 1.0))
            for k, v in weights.items()
        }
        total = sum(adjusted.values())
        if total <= 0:
            return weights
        return {k: v / total for k, v in adjusted.items()}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _softmax(values: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """"""
    keys = list(values.keys())
    scaled = [values[k] / temperature for k in keys]
    max_v = max(scaled)
    exp_v = [math.exp(v - max_v) for v in scaled]
    total = sum(exp_v)
    return {k: e / total for k, e in zip(keys, exp_v)}


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _std(lst: List[float]) -> float:
    if len(lst) < 2:
        return 0.0
    mu = _mean(lst)
    var = sum((x - mu) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(var) if var > 0 else 0.0
