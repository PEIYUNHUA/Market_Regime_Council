"""
PPO — Proximal Policy Optimization baseline.

Identical actor-critic structure to A2C but with the PPO clipped objective:
  L_CLIP = −min(ρ·A,  clip(ρ, 1−ε, 1+ε)·A)
where ρ = π_θ(a|s) / π_θ_old(a|s)  is the probability ratio.

Old policy is snapshotted before each update and used to compute ρ.
Clip parameter ε = 0.2 (standard PPO default).

Reference:
  Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
  Used as DRL baseline in FinCon (NeurIPS 2024) Appendix B.
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


class PPOBaseline(_DRLBase):
    """
    Proximal Policy Optimization baseline.

    Identical actor-critic structure to A2C but with the PPO clipped objective:
      L_CLIP = −min(ρ·A,  clip(ρ, 1−ε, 1+ε)·A)
    where ρ = π_θ(a|s) / π_θ_old(a|s)  is the probability ratio.

    Old policy is snapshotted before each update and used to compute ρ.
    Clip parameter ε = 0.2 (standard PPO default).
    """

    CLIP_EPS = 0.2

    def __init__(
        self,
        lookback:     int   = LOOKBACK,
        lr_actor:     float = 3e-4,
        lr_critic:    float = 6e-4,
        entropy_coef: float = 0.01,
        seed:         int   = 42,
    ):
        super().__init__(lookback=lookback, seed=seed)
        self.entropy_coef = entropy_coef

        self._actor  = _MLP(self.feat_dim, 64, self.n_out, lr=lr_actor,  seed=seed)
        self._critic = _MLP(self.feat_dim, 64, 1,          lr=lr_critic, seed=seed + 1)

        self._old_actor_w: Optional[List[np.ndarray]] = None

        self._last_s:      Optional[np.ndarray] = None
        self._last_logits: Optional[np.ndarray] = None
        self._last_V:      Optional[float] = None
        self._n_updates:   int = 0

        logger.info("[PPO] Initialized | feat_dim=%d ε=%.2f lr_actor=%.4f",
                    self.feat_dim, self.CLIP_EPS, lr_actor)

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        self._price_buf.append(dict(bundle.prices))
        self._n_periods += 1

        s = self._features()
        if s is None:
            return self._fallback_equal()

        # Snapshot old policy before forward pass
        self._old_actor_w = self._actor.get_weights()

        logits = self._actor.forward(s)
        V      = float(self._critic.forward(s))

        self._last_s      = s
        self._last_logits = logits.copy()
        self._last_V      = V

        tok_w, cash_w = self._to_portfolio(logits)
        logger.debug("[PPO] %s | n=%d V̂=%.4f cash=%.2f",
                     bundle.eval_date, self._n_updates, V, cash_w)

        return {"portfolio_weights": tok_w, "cash_weight": cash_w,
                "market_regime": "ppo"}

    def _on_reward(self, r: float):
        if self._last_s is None or self._old_actor_w is None:
            return

        s, logits, V = self._last_s, self._last_logits, self._last_V
        A = r - V                              # advantage

        # ── Critic update (same as A2C) ───────────────────────────────────
        self._critic.forward(s)
        self._critic.backward_update(np.array([2.0 * (V - r)]))

        # ── PPO clipped actor update ──────────────────────────────────────
        #   New policy logits
        new_logits = self._actor.forward(s)
        new_w = _softmax(new_logits)

        #   Old policy logits (frozen snapshot)
        old_w_all = self._actor.get_weights()   # save to restore later
        self._actor.set_weights(self._old_actor_w)
        old_logits = self._actor.forward(s)
        old_w = _softmax(old_logits)
        self._actor.set_weights(old_w_all)      # restore current policy

        # Probability ratio at argmax action (categorical approximation)
        a_idx = int(np.argmax(new_w))
        π_new = new_w[a_idx]
        π_old = old_w[a_idx]
        ρ = (π_new / (π_old + 1e-10))

        # Clipped surrogate
        ρ_clip = np.clip(ρ, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS)
        # −min(ρ·A, ρ_clip·A): gradient with respect to ρ
        # If A > 0: min = ρ·A when ρ < ρ_clip  →  used_ρ = ρ  →  grad = −A (when ρ < 1+ε)
        #           else min = ρ_clip·A  →  no gradient (clipped)
        used_ρ = ρ if (A >= 0 and ρ < 1 + self.CLIP_EPS) or \
                     (A < 0  and ρ > 1 - self.CLIP_EPS) else ρ_clip
        # Gradient of −used_ρ·A w.r.t. logits (via log π):
        #   d(−ρ·A)/d logits = −A · ρ · d log π / d logits
        #                     = −A · ρ · (e_a − w)
        e_a = np.zeros(self.n_out)
        e_a[a_idx] = 1.0
        pg_grad  = -A * used_ρ * (e_a - new_w)

        # Entropy bonus
        ent_grad = self.entropy_coef * (np.log(new_w + 1e-8) + 1.0)

        self._actor.forward(s)
        self._actor.backward_update(pg_grad + ent_grad)

        self._n_updates += 1

    def _fallback_equal(self) -> Dict[str, Any]:
        w = {tok: 1.0 / self.n_tokens for tok in self.tokens}
        return {"portfolio_weights": w, "cash_weight": 0.0, "market_regime": "ppo_warmup"}
