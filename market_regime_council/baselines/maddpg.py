"""
MADDPG — Multi-Agent Deep Deterministic Policy Gradient baseline.

Cooperative variant: all 3 agents maximise joint portfolio return.
Each agent has a deterministic actor and a centralised critic:
  Actor_i:  MLP(91 → 64 → 14), deterministic, outputs softmax portfolio weights
  Critic_i: MLP(260 + 14*3 → 128 → 1), centralised — sees full_obs + all agents' actions

Temporal feature split (same as MAPPO):
  Agent 0 (short-term):  last 7 days   → 91-dim slice
  Agent 1 (medium-term): days -14 to -7 → 91-dim slice
  Agent 2 (long-term):   days 0 to -14  → 78-dim, zero-padded to 91

Target networks (soft sync τ=0.005) stabilise training.
Shared replay buffer (deque, maxlen=512).
Final portfolio: average of 3 agents' softmax outputs.

Reference:
  Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
  Environments", NeurIPS 2017.
  Cooperative variant applied to crypto portfolio management in multiple
  2024 papers.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._drl_base import (
    _MLP, _DRLBase, _softmax,
    TOKENS, N_TOKENS, N_OUT, LOOKBACK, FEATURE_DIM,
    DateRangeBundle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  MADDPG constants
# ---------------------------------------------------------------------------

N_AGENTS    = 3
FEAT_PER_AGENT = 91          # per-agent temporal slice dimension
CRITIC_IN   = FEATURE_DIM + N_OUT * N_AGENTS   # 260 + 42 = 302
GAMMA       = 0.99
TAU         = 0.005          # soft target sync rate
BATCH_SIZE  = 32
BUFFER_CAP  = 512
ENTROPY_COEF = 0.005         # lighter entropy than single-agent methods


def _extract_agent_slices(full_feat: np.ndarray) -> List[np.ndarray]:
    """
    Split the 260-dim feature vector (13 tokens × 20 days) into 3 temporal slices.
    Identical layout to MAPPO's _extract_agent_slices.
    """
    mat = full_feat.reshape(N_TOKENS, LOOKBACK)  # (13, 20)

    # Agent 0: short-term — last 7 days
    s0 = mat[:, -7:].flatten()                    # (91,)

    # Agent 1: medium-term — days -14 to -7
    s1 = mat[:, -14:-7].flatten()                 # (91,)

    # Agent 2: long-term — days 0 to -14
    s2_raw = mat[:, :-14].flatten()               # (78,)
    s2 = np.zeros(FEAT_PER_AGENT, dtype=np.float32)
    s2[:len(s2_raw)] = s2_raw

    return [s0, s1, s2]


def _soft_sync(target: _MLP, source: _MLP, tau: float = TAU):
    """Polyak / soft target update: θ_target ← τ·θ_source + (1−τ)·θ_target."""
    tw = target.get_weights()
    sw = source.get_weights()
    new_w = [tau * s + (1.0 - tau) * t for t, s in zip(tw, sw)]
    target.set_weights(new_w)


class MADDPGBaseline(_DRLBase):
    """
    Multi-Agent DDPG baseline (cooperative variant, all agents share reward).

    N_AGENTS = 3 temporal-specialised agents.
    Centralised critic sees full observation + all agents' actions concatenated.
    Final portfolio = average of 3 agents' softmax outputs.
    """

    def __init__(
        self,
        lookback:   int   = LOOKBACK,   # ignored — MADDPG uses its own fixed LOOKBACK=20
        lr_actor:   float = 1e-4,
        lr_critic:  float = 3e-4,
        seed:       int   = 42,
    ):
        # Always use the fixed LOOKBACK=20 for temporal feature slicing.
        # The external `lookback` parameter (default 90 in backtest) would produce
        # a 1170-dim feature vector that cannot be reshaped to (13, 20).
        super().__init__(lookback=LOOKBACK, seed=seed)

        # Actors and their targets
        self._actors:         List[_MLP] = [
            _MLP(FEAT_PER_AGENT, 64, self.n_out, lr=lr_actor,  seed=seed + i)
            for i in range(N_AGENTS)
        ]
        self._target_actors:  List[_MLP] = [
            _MLP(FEAT_PER_AGENT, 64, self.n_out, lr=lr_actor,  seed=seed + i)
            for i in range(N_AGENTS)
        ]

        # Critics and their targets (centralised: full_obs + all actions)
        self._critics:        List[_MLP] = [
            _MLP(CRITIC_IN, 128, 1, lr=lr_critic, seed=seed + 20 + i)
            for i in range(N_AGENTS)
        ]
        self._target_critics: List[_MLP] = [
            _MLP(CRITIC_IN, 128, 1, lr=lr_critic, seed=seed + 20 + i)
            for i in range(N_AGENTS)
        ]

        # Initialise targets to match current networks
        for i in range(N_AGENTS):
            self._target_actors[i].set_weights(self._actors[i].get_weights())
            self._target_critics[i].set_weights(self._critics[i].get_weights())

        # Shared replay buffer: stores (full_obs, [a0,a1,a2], r, full_obs_next)
        self._replay: deque = deque(maxlen=BUFFER_CAP)

        # Transient state for one period
        self._last_full_feat: Optional[np.ndarray] = None
        self._last_actions:   Optional[List[np.ndarray]] = None   # softmax outputs
        self._n_updates:      int = 0

        logger.info(
            "[MADDPG] Initialized | n_agents=%d feat_per_agent=%d "
            "lr_actor=%.0e lr_critic=%.0e",
            N_AGENTS, FEAT_PER_AGENT, lr_actor, lr_critic,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _actions_concat(self, actions: List[np.ndarray]) -> np.ndarray:
        """Concatenate all agents' action vectors into one flat array (42 dims)."""
        return np.concatenate(actions)

    def _critic_input(
        self,
        full_feat: np.ndarray,
        actions: List[np.ndarray],
    ) -> np.ndarray:
        """Build centralised critic input: [full_obs, a0, a1, a2]."""
        return np.concatenate([full_feat, self._actions_concat(actions)])

    # ------------------------------------------------------------------
    #  Interface
    # ------------------------------------------------------------------

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        self._price_buf.append(dict(bundle.prices))
        self._n_periods += 1

        full_feat = self._features()
        if full_feat is None:
            return self._fallback_equal()

        slices = _extract_agent_slices(full_feat)

        # Forward each actor → softmax action
        actions = [
            _softmax(self._actors[i].forward(slices[i]))
            for i in range(N_AGENTS)
        ]

        # Store for training
        self._last_full_feat = full_feat
        self._last_actions   = [a.copy() for a in actions]

        # Final portfolio: average of all agents' action distributions
        avg_weights = np.mean(np.stack(actions, axis=0), axis=0)
        tok_w, cash_w = self._to_portfolio(avg_weights)

        logger.debug("[MADDPG] %s | n=%d cash=%.2f",
                     bundle.eval_date, self._n_updates, cash_w)

        return {"portfolio_weights": tok_w, "cash_weight": cash_w,
                "market_regime": "maddpg"}

    def _on_reward(self, r: float):
        if self._last_full_feat is None or self._last_actions is None:
            return

        prev_feat = self._last_full_feat
        prev_acts = self._last_actions

        # Current state features (already updated by run_with_bundle's price append)
        curr_feat = self._features()
        if curr_feat is None:
            return

        # Store transition
        self._replay.append((
            prev_feat.copy(),
            [a.copy() for a in prev_acts],
            r,
            curr_feat.copy(),
        ))

        if len(self._replay) < BATCH_SIZE:
            return

        # Sample mini-batch
        batch = random.sample(self._replay, BATCH_SIZE)

        # ── Per-sample update (simplified online variant) ──────────────
        for obs, acts, reward, obs_next in batch:
            slices_next = _extract_agent_slices(obs_next)

            # Target actions from target actors
            tgt_acts = [
                _softmax(self._target_actors[i].forward(slices_next[i]))
                for i in range(N_AGENTS)
            ]

            # ── Critic update for each agent ──────────────────────────
            for i in range(N_AGENTS):
                # Target Q value
                c_in_next = self._critic_input(obs_next, tgt_acts)
                q_next    = float(self._target_critics[i].forward(c_in_next))
                y         = reward + GAMMA * q_next

                # Current Q value
                c_in      = self._critic_input(obs, acts)
                q_pred    = float(self._critics[i].forward(c_in))

                # Huber loss gradient
                td_err    = q_pred - y
                huber_g   = np.sign(td_err) if abs(td_err) > 1.0 else td_err
                # critic input dimension = CRITIC_IN, output = 1
                self._critics[i].forward(c_in)
                self._critics[i].backward_update(np.array([huber_g / BATCH_SIZE]))

            # ── Actor update (REINFORCE approximation of DPG) ─────────
            slices_curr = _extract_agent_slices(obs)
            curr_acts   = [
                _softmax(self._actors[i].forward(slices_curr[i]))
                for i in range(N_AGENTS)
            ]

            for i in range(N_AGENTS):
                # Estimate Q from current critic using current actions
                c_in_curr = self._critic_input(obs, curr_acts)
                q_val     = float(self._critics[i].forward(c_in_curr))

                # Actor gradient: -Q_value * (e_argmax - softmax(logits_i))
                # (treats Q as advantage signal analogously to DPG)
                logits_i = self._actors[i].forward(slices_curr[i])
                w_i      = _softmax(logits_i)
                a_idx    = int(np.argmax(w_i))
                e_a      = np.zeros(self.n_out)
                e_a[a_idx] = 1.0

                # Negative because we maximise Q
                pg_grad  = -q_val * (e_a - w_i) / BATCH_SIZE

                # Entropy bonus (lighter than single-agent; encourages diversity)
                ent_grad = ENTROPY_COEF * (np.log(w_i + 1e-8) + 1.0)

                self._actors[i].forward(slices_curr[i])
                self._actors[i].backward_update(pg_grad + ent_grad)

        # ── Soft target sync ──────────────────────────────────────────
        for i in range(N_AGENTS):
            _soft_sync(self._target_actors[i],  self._actors[i],  TAU)
            _soft_sync(self._target_critics[i], self._critics[i], TAU)

        self._n_updates += 1

    def _fallback_equal(self) -> Dict[str, Any]:
        w = {tok: 1.0 / self.n_tokens for tok in self.tokens}
        return {"portfolio_weights": w, "cash_weight": 0.0, "market_regime": "maddpg_warmup"}
