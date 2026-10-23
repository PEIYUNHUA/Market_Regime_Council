"""
MAPPO — Multi-Agent Proximal Policy Optimization baseline.

Three cooperative agents, each specialising on a different temporal horizon:
  Agent 0 (short-term):  last 7 days   → 91-dim feature slice
  Agent 1 (medium-term): days -14 to -7 → 91-dim feature slice
  Agent 2 (long-term):   days 0 to -14  → 78-dim, zero-padded to 91

Each agent has its own Actor MLP(91→64→14).
A shared centralised critic MLP(260→128→1) sees the full observation.
Final portfolio: average of three agents' softmax outputs → _to_portfolio.

Training (called once per period after realised return):
  Advantage A = r − V(full_obs)
  Critic update: grad = 2*(V−r)
  Each actor: PPO clipping ε=0.2, entropy bonus coef=0.01

Reference:
  Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games", NeurIPS Deep RL Workshop 2021.
  Applied to crypto portfolio management in multiple 2023-2025 finance papers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ._drl_base import (
    _MLP, _DRLBase, _softmax,
    TOKENS, N_TOKENS, N_OUT, LOOKBACK, FEATURE_DIM,
    DateRangeBundle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  MAPPO constants
# ---------------------------------------------------------------------------

N_AGENTS = 3
FEAT_PER_AGENT = 91   # 13 tokens × 7 days (short/medium), zero-padded for long
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01


def _extract_agent_slices(full_feat: np.ndarray) -> List[np.ndarray]:
    """
    Split the 260-dim feature vector (13 tokens × 20 days) into 3 temporal slices.

    The features are laid out as:
      [tok0_day0, tok0_day1, ..., tok0_day19,
       tok1_day0, ..., tok1_day19,
       ...
       tok12_day0, ..., tok12_day19]

    Reshape to (13, 20) then slice along the time axis:
      Agent 0: days[-7:]   → shape (13, 7) → flatten → 91 dims
      Agent 1: days[-14:-7]→ shape (13, 7) → flatten → 91 dims
      Agent 2: days[:-14]  → shape (13, 6) → flatten → 78 dims, zero-pad to 91
    """
    mat = full_feat.reshape(N_TOKENS, LOOKBACK)  # (13, 20)

    # Agent 0: short-term — last 7 days
    s0 = mat[:, -7:].flatten()                    # (91,)

    # Agent 1: medium-term — days -14 to -7
    s1 = mat[:, -14:-7].flatten()                 # (91,)

    # Agent 2: long-term — days 0 to -14 (first 6 days)
    s2_raw = mat[:, :-14].flatten()               # (78,)
    s2 = np.zeros(FEAT_PER_AGENT, dtype=np.float32)
    s2[:len(s2_raw)] = s2_raw                     # zero-pad to 91

    return [s0, s1, s2]


class MAPPOBaseline(_DRLBase):
    """
    Multi-Agent PPO baseline with centralised critic and decentralised actors.

    N_AGENTS = 3 agents specialise on short/medium/long-term signals.
    Final portfolio = average of 3 agents' softmax outputs.
    """

    def __init__(
        self,
        lookback:     int   = LOOKBACK,   # ignored — MAPPO uses its own fixed LOOKBACK=20
        lr_actor:     float = 3e-4,
        lr_critic:    float = 6e-4,
        entropy_coef: float = ENTROPY_COEF,
        seed:         int   = 42,
    ):
        # Always use the fixed LOOKBACK=20 for temporal feature slicing.
        # The external `lookback` parameter (default 90 in backtest) would produce
        # a 1170-dim feature vector that cannot be reshaped to (13, 20).
        super().__init__(lookback=LOOKBACK, seed=seed)
        self.entropy_coef = entropy_coef

        # Each agent has its own actor; different seeds for diversity
        self._actors: List[_MLP] = [
            _MLP(FEAT_PER_AGENT, 64, self.n_out, lr=lr_actor, seed=seed + i)
            for i in range(N_AGENTS)
        ]

        # Centralised critic sees full 260-dim observation
        self._critic = _MLP(FEATURE_DIM, 128, 1, lr=lr_critic, seed=seed + 10)

        # Storage for update
        self._last_full_feat: Optional[np.ndarray] = None
        self._last_slices:    Optional[List[np.ndarray]] = None
        self._last_logits:    Optional[List[np.ndarray]] = None
        self._last_V:         Optional[float] = None

        # Old policy snapshots for PPO ratio
        self._old_actor_ws: Optional[List[List[np.ndarray]]] = None

        self._n_updates: int = 0

        logger.info(
            "[MAPPO] Initialized | n_agents=%d feat_per_agent=%d lr_actor=%.0e",
            N_AGENTS, FEAT_PER_AGENT, lr_actor,
        )

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        self._price_buf.append(dict(bundle.prices))
        self._n_periods += 1

        full_feat = self._features()
        if full_feat is None:
            return self._fallback_equal()

        slices = _extract_agent_slices(full_feat)

        # Snapshot old policies before forward pass
        self._old_actor_ws = [actor.get_weights() for actor in self._actors]

        # Forward each actor and centralised critic
        logits_list = [self._actors[i].forward(slices[i]) for i in range(N_AGENTS)]
        V = float(self._critic.forward(full_feat))

        # Store state for training
        self._last_full_feat = full_feat
        self._last_slices    = slices
        self._last_logits    = [lg.copy() for lg in logits_list]
        self._last_V         = V

        # Final portfolio: average of 3 agents' softmax outputs
        avg_logits = np.mean(np.stack([_softmax(lg) for lg in logits_list], axis=0), axis=0)
        tok_w, cash_w = self._to_portfolio(avg_logits)

        logger.debug("[MAPPO] %s | n=%d V̂=%.4f cash=%.2f",
                     bundle.eval_date, self._n_updates, V, cash_w)

        return {"portfolio_weights": tok_w, "cash_weight": cash_w,
                "market_regime": "mappo"}

    def _on_reward(self, r: float):
        if self._last_full_feat is None or self._old_actor_ws is None:
            return

        full_feat  = self._last_full_feat
        slices     = self._last_slices
        logits_list = self._last_logits
        V          = self._last_V
        A          = r - V                             # advantage

        # ── Centralised critic update: minimise (r − V)² ──────────────────
        self._critic.forward(full_feat)
        self._critic.backward_update(np.array([2.0 * (V - r)]))

        # ── Each actor: PPO clipped update ────────────────────────────────
        for i in range(N_AGENTS):
            actor      = self._actors[i]
            s_i        = slices[i]
            old_logits = logits_list[i]

            # New policy
            new_logits = actor.forward(s_i)
            new_w      = _softmax(new_logits)

            # Old policy (frozen snapshot)
            curr_ws = actor.get_weights()
            actor.set_weights(self._old_actor_ws[i])
            old_w = _softmax(actor.forward(s_i))
            actor.set_weights(curr_ws)               # restore

            # Probability ratio at argmax action
            a_idx = int(np.argmax(new_w))
            π_new = new_w[a_idx]
            π_old = old_w[a_idx]
            ρ = π_new / (π_old + 1e-10)

            # Clipped surrogate
            ρ_clip = np.clip(ρ, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            used_ρ = ρ if (A >= 0 and ρ < 1 + CLIP_EPS) or \
                         (A < 0  and ρ > 1 - CLIP_EPS) else ρ_clip

            e_a = np.zeros(self.n_out)
            e_a[a_idx] = 1.0
            pg_grad  = -A * used_ρ * (e_a - new_w)

            # Entropy bonus
            ent_grad = self.entropy_coef * (np.log(new_w + 1e-8) + 1.0)

            actor.forward(s_i)
            actor.backward_update(pg_grad + ent_grad)

        self._n_updates += 1

    def _fallback_equal(self) -> Dict[str, Any]:
        w = {tok: 1.0 / self.n_tokens for tok in self.tokens}
        return {"portfolio_weights": w, "cash_weight": 0.0, "market_regime": "mappo_warmup"}
