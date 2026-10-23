"""
A2C — Advantage Actor-Critic baseline.

Actor:   w = softmax(Actor_MLP(s))           — portfolio allocation
Critic:  V̂ = Critic_MLP(s)                  — expected return estimate
Advantage: A = r − V̂(s)
Actor loss:  −A · Σᵢ wᵢ · ∇θ log wᵢ  (REINFORCE with baseline)
Critic loss: (r − V̂)²
Entropy bonus coefficient: 0.01

Reference:
  Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016.
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


class A2CBaseline(_DRLBase):
    """
    Advantage Actor-Critic baseline.

    Actor:   w = softmax(Actor_MLP(s))           — portfolio allocation
    Critic:  V̂ = Critic_MLP(s)                  — expected return estimate
    Advantage: A = r − V̂(s)
    Actor loss:  −A · Σᵢ wᵢ · ∇θ log wᵢ  (REINFORCE with baseline)
    Critic loss: (r − V̂)²
    Entropy bonus coefficient: 0.01
    """

    def __init__(
        self,
        lookback:      int   = LOOKBACK,
        lr_actor:      float = 1e-3,
        lr_critic:     float = 2e-3,
        entropy_coef:  float = 0.01,
        seed:          int   = 42,
    ):
        super().__init__(lookback=lookback, seed=seed)
        self.entropy_coef = entropy_coef

        self._actor  = _MLP(self.feat_dim, 64, self.n_out,  lr=lr_actor,  seed=seed)
        self._critic = _MLP(self.feat_dim, 64, 1,           lr=lr_critic, seed=seed + 1)

        self._last_s:    Optional[np.ndarray] = None
        self._last_logits: Optional[np.ndarray] = None
        self._last_V:    Optional[float] = None
        self._n_updates: int = 0

        logger.info("[A2C] Initialized | feat_dim=%d lr_actor=%.4f lr_critic=%.4f",
                    self.feat_dim, lr_actor, lr_critic)

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        self._price_buf.append(dict(bundle.prices))
        self._n_periods += 1

        s = self._features()
        if s is None:
            return self._fallback_equal()

        logits = self._actor.forward(s)
        V      = float(self._critic.forward(s))

        self._last_s      = s
        self._last_logits = logits.copy()
        self._last_V      = V

        tok_w, cash_w = self._to_portfolio(logits)
        logger.debug("[A2C] %s | n=%d V̂=%.4f cash=%.2f",
                     bundle.eval_date, self._n_updates, V, cash_w)

        return {"portfolio_weights": tok_w, "cash_weight": cash_w,
                "market_regime": "a2c"}

    def _on_reward(self, r: float):
        if self._last_s is None:
            return

        s, logits, V = self._last_s, self._last_logits, self._last_V
        A = r - V                             # advantage

        # ── Critic update: minimise (r − V)² ──────────────────────────────
        self._critic.forward(s)
        self._critic.backward_update(np.array([2.0 * (V - r)]))   # grad of (V-r)²

        # ── Actor update: REINFORCE with advantage + entropy bonus ─────────
        #   Policy gradient direction: ∇ log π(w|s) = ∂/∂θ log softmax
        #   For softmax: ∇ log softmax_i = δ_ij − w_j   (Jacobian diagonal)
        #   With REINFORCE: grad = −A · (1[best] − w)   [treat argmax as action]
        #   Entropy: H(w) = −Σ wᵢ log wᵢ;  ∇_θ H = −w · (log w + 1) · J_softmax
        w = _softmax(logits)                           # current portfolio

        # Policy gradient: push toward high-weight tokens if A>0, else away
        # Using REINFORCE categorical approximation at argmax(w):
        e_a = np.zeros(self.n_out)
        e_a[int(np.argmax(w))] = 1.0
        pg_grad = -A * (e_a - w)                      # grad of −A·log π

        # Entropy gradient (pre-softmax): encourages exploration
        ent_grad = self.entropy_coef * (np.log(w + 1e-8) + 1.0)

        self._actor.forward(s)
        self._actor.backward_update(pg_grad + ent_grad)

        self._n_updates += 1

    def _fallback_equal(self) -> Dict[str, Any]:
        w = {tok: 1.0 / self.n_tokens for tok in self.tokens}
        return {"portfolio_weights": w, "cash_weight": 0.0, "market_regime": "a2c_warmup"}
