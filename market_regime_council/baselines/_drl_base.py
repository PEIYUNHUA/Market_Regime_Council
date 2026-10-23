"""
Shared infrastructure for DRL-based trading baselines.

Provides:
  Constants:  TOKENS, N_TOKENS, N_OUT, LOOKBACK, FEATURE_DIM
  Helpers:    _softmax(), _relu()
  Classes:    _MLP (2-layer MLP with Adam), _DRLBase (common interface)

All DRL algorithm modules (a2c, ppo, dqn, mappo, maddpg) import from here.
No log messages are emitted from this module — each algorithm file defines
its own logger = logging.getLogger(__name__).
"""
from __future__ import annotations

import datetime
import logging
import math
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import ALL_TOKENS
from ..council.regime_council import DateRangeBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

TOKENS: List[str] = ALL_TOKENS           # 13 tokens
N_TOKENS: int = len(TOKENS)              # 13
N_OUT: int = N_TOKENS + 1               # 13 tokens + 1 cash = 14
LOOKBACK: int = 20                       # days of return history as features
FEATURE_DIM: int = N_TOKENS * LOOKBACK  # 260

# ---------------------------------------------------------------------------
#  Numerical utilities
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-12)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ---------------------------------------------------------------------------
#  Minimal 2-layer MLP with Adam optimiser (NumPy-only)
# ---------------------------------------------------------------------------


class _MLP:
    """
    Two-layer fully-connected network:
      output = W2 · ReLU(W1 · input + b1) + b2
    Optimised with Adam.  Supports batched forward; single-sample backward.
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 lr: float = 1e-3, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim,  hid_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hid_dim)
        self.W2 = rng.randn(hid_dim, out_dim) * np.sqrt(2.0 / hid_dim)
        self.b2 = np.zeros(out_dim)
        self.lr = lr

        # Adam state (β1=0.9, β2=0.999)
        self._β1, self._β2, self._ε = 0.9, 0.999, 1e-8
        self._t = 0
        self._m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]

    # ---- forward ----

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (in_dim,) — caches activations for backward."""
        self._x = x
        self._h = _relu(x @ self.W1 + self.b1)
        return self._h @ self.W2 + self.b2

    # ---- backward + update ----

    def backward_update(self, grad_out: np.ndarray):
        """
        grad_out: gradient of scalar loss wrt network output, shape (out_dim,).
        Runs one Adam step.
        """
        dW2 = np.outer(self._h, grad_out)
        db2 = grad_out
        dh  = (grad_out @ self.W2.T) * (self._h > 0)
        dW1 = np.outer(self._x, dh)
        db1 = dh

        self._t += 1
        params  = [self.W1, self.b1, self.W2, self.b2]
        grads   = [dW1,     db1,     dW2,     db2]
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self._β1 * self._m[i] + (1 - self._β1) * g
            self._v[i] = self._β2 * self._v[i] + (1 - self._β2) * g ** 2
            m̂ = self._m[i] / (1 - self._β1 ** self._t)
            v̂ = self._v[i] / (1 - self._β2 ** self._t)
            p -= self.lr * m̂ / (np.sqrt(v̂) + self._ε)

    # ---- snapshot for PPO old-policy ----

    def get_weights(self) -> List[np.ndarray]:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_weights(self, ws: List[np.ndarray]):
        self.W1[:] = ws[0]; self.b1[:] = ws[1]
        self.W2[:] = ws[2]; self.b2[:] = ws[3]


# ---------------------------------------------------------------------------
#  Shared base for all DRL baselines
# ---------------------------------------------------------------------------


class _DRLBase:
    """
    Common infrastructure: price history ring-buffer, feature extraction,
    weight normalisation, and the run_with_bundle / record_period_result interface.
    """

    MAX_TOKEN_W = 0.35    # single-token cap
    MAX_CASH_W  = 0.30    # cash cap

    def __init__(self, lookback: int = LOOKBACK, seed: int = 42):
        self.tokens     = TOKENS
        self.n_tokens   = N_TOKENS
        self.n_out      = N_OUT
        self.lookback   = lookback
        self.feat_dim   = N_TOKENS * lookback

        self._price_buf: deque = deque(maxlen=lookback + 2)
        self._n_periods: int = 0

        random.seed(seed)
        np.random.seed(seed)

    # ---- feature extraction ----

    def _features(self) -> Optional[np.ndarray]:
        """
        Returns normalised feature vector of shape (feat_dim,), or None during
        the initial warm-up when fewer than LOOKBACK + 1 price observations exist.
        """
        if len(self._price_buf) < 2:
            return None

        hist = list(self._price_buf)
        raw: List[float] = []
        for tok in self.tokens:
            prices = [h.get(tok, 0.0) for h in hist]
            rets = []
            for j in range(1, len(prices)):
                p0, p1 = prices[j - 1], prices[j]
                rets.append(math.log(p1 / p0) if p0 > 0 and p1 > 0 else 0.0)
            # Pad or trim to exactly LOOKBACK values
            while len(rets) < self.lookback:
                rets.insert(0, 0.0)
            raw.extend(rets[-self.lookback:])

        feat = np.array(raw, dtype=np.float32)
        σ = feat.std()
        if σ > 1e-9:
            feat /= σ
        return np.clip(feat, -5.0, 5.0)

    # ---- weight conversion ----

    def _to_portfolio(
        self, logits: np.ndarray
    ) -> Tuple[Dict[str, float], float]:
        """
        softmax(logits) → (token_weights dict, cash_weight)
        Enforces per-token cap and cash cap, then re-normalises.
        """
        w = _softmax(logits)           # shape (N_OUT,)  last element = cash
        tok_w = {tok: float(w[i]) for i, tok in enumerate(self.tokens)}
        cash_w = float(w[-1])

        # Enforce caps
        for tok in self.tokens:
            if tok_w[tok] > self.MAX_TOKEN_W:
                cash_w += tok_w[tok] - self.MAX_TOKEN_W
                tok_w[tok] = self.MAX_TOKEN_W
        cash_w = min(cash_w, self.MAX_CASH_W)

        # Re-normalise
        total = sum(tok_w.values()) + cash_w
        if total > 1e-9:
            tok_w  = {k: v / total for k, v in tok_w.items()}
            cash_w /= total

        return tok_w, cash_w

    # ---- interface ----

    async def run_with_bundle(self, bundle: DateRangeBundle) -> Dict[str, Any]:
        raise NotImplementedError

    def record_period_result(
        self,
        date: datetime.date,
        actual_return: float,
        portfolio_value: float,
        prediction: Optional[Dict[str, Any]] = None,
    ):
        self._on_reward(actual_return)

    def _on_reward(self, reward: float):
        pass
