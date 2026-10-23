"""
DQN — Deep Q-Network baseline with discretised portfolio action space.

Q-network:   Q(s, a) for a ∈ {10 predefined portfolio templates}
Exploration: ε-greedy, ε decays from 1.0 → 0.05 over 300 steps
Replay:      RandomSampling replay buffer (capacity=512)
Target net:  synchronised every 30 update steps
Discount:    γ = 0.99
Loss:        Huber loss (δ=1) between Q(s,a) and r + γ max_a' Q_target(s',a')

Reference:
  Mnih et al., "Human-level control through deep reinforcement learning",
  Nature 2015.
  Used as DRL baseline in FinCon (NeurIPS 2024) Appendix B.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from ._drl_base import (
    _MLP, _DRLBase, _softmax,
    TOKENS, N_TOKENS, N_OUT, LOOKBACK, FEATURE_DIM,
    DateRangeBundle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  10 predefined portfolio templates — the discrete action space for DQN.
#  Each template is a dict mapping token → weight fraction (cash = 1 − Σ).
# ---------------------------------------------------------------------------


def _build_templates() -> List[Dict[str, float]]:
    toks = TOKENS
    N = len(toks)
    eq = 1.0 / N

    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        """Normalise token weights to sum ≤ 1 (remainder becomes cash)."""
        t = sum(d.values())
        return {k: v / t for k, v in d.items()} if t > 1.0 else d

    templates = [
        # 0: equal-weight (no cash)
        {tok: eq for tok in toks},
        # 1: BTC-heavy (30% BTC, rest equal)
        {**{t: (1.0 - 0.30) / (N - 1) for t in toks if t != "BTC"}, "BTC": 0.30},
        # 2: ETH-heavy (25% ETH, rest equal)
        {**{t: (1.0 - 0.25) / (N - 1) for t in toks if t != "ETH"}, "ETH": 0.25},
        # 3: large-cap focus (BTC 25%, ETH 20%, BNB 15%, rest equal 10)
        _norm({**{t: (1.0 - 0.60) / (N - 3) for t in toks
                  if t not in ("BTC", "ETH", "BNB")},
               "BTC": 0.25, "ETH": 0.20, "BNB": 0.15}),
        # 4: cash-10% (10% cash, rest equal)
        {tok: 0.90 * eq for tok in toks},   # cash_w = 0.10
        # 5: cash-25% (25% cash, rest equal)
        {tok: 0.75 * eq for tok in toks},   # cash_w = 0.25
        # 6: cash-50% (50% cash, rest equal)
        {tok: 0.50 * eq for tok in toks},   # cash_w = 0.50 (capped to 0.30 by _to_portfolio)
        # 7: alt-coin focus (uniform over 10 non-BTC/ETH/BNB)
        {tok: (1.0 / (N - 3)) for tok in toks if tok not in ("BTC", "ETH", "BNB")},
        # 8: safe-haven (PAXG + BTC + LTC + ZEC equal split, no other tokens)
        {"PAXG": 0.25, "BTC": 0.25, "LTC": 0.25, "ZEC": 0.25},
        # 9: defensive (30% cash, 30% PAXG, 40% equal over remaining 12 tokens)
        {**{tok: 0.40 / (N - 1) for tok in toks if tok != "PAXG"},
         "PAXG": 0.30},   # cash_w = 0.30
    ]
    return templates


_PORTFOLIO_TEMPLATES = _build_templates()
N_DISCRETE = len(_PORTFOLIO_TEMPLATES)   # 10


def _template_cash(tmpl: Dict[str, float]) -> float:
    """Infer cash weight = 1 − Σ token weights."""
    return max(0.0, 1.0 - sum(tmpl.values()))


class DQNBaseline(_DRLBase):
    """
    Deep Q-Network baseline with discretised portfolio action space.

    Q-network:   Q(s, a) for a ∈ {10 predefined portfolio templates}
    Exploration: ε-greedy, ε decays from 1.0 → 0.05 over 300 steps
    Replay:      RandomSampling replay buffer (capacity=512)
    Target net:  synchronised every 30 update steps
    Discount:    γ = 0.99
    Loss:        Huber loss (δ=1) between Q(s,a) and r + γ max_a' Q_target(s',a')
    """

    BUFFER_CAP  = 512
    BATCH_SIZE  = 32
    GAMMA       = 0.99
    TARGET_SYNC = 30
    EPS_START   = 1.0
    EPS_END     = 0.05
    EPS_DECAY   = 300   # linear decay over this many steps

    def __init__(
        self,
        lookback: int   = LOOKBACK,
        lr:       float = 5e-4,
        seed:     int   = 42,
    ):
        super().__init__(lookback=lookback, seed=seed)

        self._qnet    = _MLP(self.feat_dim, 64, N_DISCRETE, lr=lr, seed=seed)
        self._target  = _MLP(self.feat_dim, 64, N_DISCRETE, lr=lr, seed=seed)
        self._sync_target()

        self._replay: deque = deque(maxlen=self.BUFFER_CAP)
        self._epsilon: float = self.EPS_START

        self._last_s:  Optional[np.ndarray] = None
        self._last_a:  int = 0
        self._last_V:  float = 0.0   # unused; kept for interface parity
        self._n_updates: int = 0
        self._n_steps:   int = 0

        logger.info("[DQN] Initialized | feat_dim=%d n_actions=%d lr=%.4f",
                    self.feat_dim, N_DISCRETE, lr)

    def _sync_target(self):
        self._target.set_weights(self._qnet.get_weights())

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        self._price_buf.append(dict(bundle.prices))
        self._n_periods += 1

        s = self._features()
        if s is None:
            return self._fallback_equal()

        # ε-greedy action selection
        self._epsilon = max(
            self.EPS_END,
            self.EPS_START - (self.EPS_START - self.EPS_END)
            * (self._n_steps / self.EPS_DECAY),
        )
        self._n_steps += 1

        if random.random() < self._epsilon:
            a = random.randrange(N_DISCRETE)
        else:
            q_vals = self._qnet.forward(s)
            a = int(np.argmax(q_vals))

        self._last_s = s
        self._last_a = a

        tmpl    = _PORTFOLIO_TEMPLATES[a]
        cash_w  = _template_cash(tmpl)
        # Enforce caps
        tok_w   = {tok: min(v, self.MAX_TOKEN_W) for tok, v in tmpl.items()}
        cash_w  = min(cash_w, self.MAX_CASH_W)
        # Re-normalise
        total   = sum(tok_w.values()) + cash_w
        if total > 1e-9:
            tok_w  = {k: v / total for k, v in tok_w.items()}
            cash_w /= total

        logger.debug("[DQN] %s | ε=%.3f a=%d cash=%.2f n=%d",
                     bundle.eval_date, self._epsilon, a, cash_w, self._n_updates)

        return {"portfolio_weights": tok_w, "cash_weight": cash_w,
                "market_regime": "dqn"}

    def _on_reward(self, r: float):
        if self._last_s is None:
            return

        s_feat = self._features()         # next-state features (already updated)
        if s_feat is None:
            return

        # Store transition (s, a, r, s')
        self._replay.append((self._last_s.copy(), self._last_a, r, s_feat.copy()))

        if len(self._replay) < self.BATCH_SIZE:
            return

        # Sample mini-batch
        batch = random.sample(self._replay, self.BATCH_SIZE)

        loss_grad = np.zeros(N_DISCRETE)
        total_loss = 0.0

        for s, a, reward, s_next in batch:
            q_pred   = self._qnet.forward(s)
            q_target = self._target.forward(s_next)
            # Bellman target
            y = reward + self.GAMMA * float(np.max(q_target))
            # Huber loss gradient at action a
            td_err = q_pred[a] - y
            huber_grad = np.sign(td_err) if abs(td_err) > 1.0 else td_err
            grad = np.zeros(N_DISCRETE)
            grad[a] = huber_grad
            # Accumulate gradient
            self._qnet.forward(s)
            self._qnet.backward_update(grad / self.BATCH_SIZE)
            total_loss += td_err ** 2

        self._n_updates += 1

        # Periodically sync target network
        if self._n_updates % self.TARGET_SYNC == 0:
            self._sync_target()
            logger.debug("[DQN] Target network synced at update %d", self._n_updates)

    def _fallback_equal(self) -> Dict[str, Any]:
        w = {tok: 1.0 / self.n_tokens for tok in self.tokens}
        return {"portfolio_weights": w, "cash_weight": 0.0, "market_regime": "dqn_warmup"}
