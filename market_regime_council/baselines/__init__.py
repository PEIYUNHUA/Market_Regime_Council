"""
Baseline trading systems for NeurIPS comparison.

LLM-based baselines (peer-reviewed):
  FinConCouncil          — FinCon (NeurIPS 2024)
  TradingAgentsCouncil   — TradingAgents (ICML 2025 Workshop)
  FinMemCouncil          — FinMem (AAAI 2024)
  CryptoTradeCouncil     — CryptoTrade (EMNLP 2024) — single-agent + reflection
  FSReasoningAgentCouncil— FS-ReasoningAgent (ICLR 2025 Workshop) — fact/subjectivity split

Single-agent DRL baselines (FinCon NeurIPS 2024 Appendix B):
  A2CBaseline          — Advantage Actor-Critic
  PPOBaseline          — Proximal Policy Optimization
  DQNBaseline          — Deep Q-Network (10 discrete portfolio templates)

Multi-agent DRL baselines:
  MAPPOBaseline        — Multi-Agent PPO (Yu et al., NeurIPS Workshop 2021; finance 2023-2025)
  MADDPGBaseline       — Multi-Agent DDPG (Lowe et al., NeurIPS 2017; cooperative variant)

All classes expose:
  run_with_bundle(bundle) -> {"portfolio_weights": {...}, "cash_weight": float, ...}
  record_period_result(date, actual_return, portfolio_value, prediction)
"""
from .fincon import FinConCouncil
from .trading_agents import TradingAgentsCouncil
from .finmem import FinMemCouncil
from .cryptotrade import CryptoTradeCouncil
from .fs_reasoning_agent import FSReasoningAgentCouncil
from .a2c import A2CBaseline
from .ppo import PPOBaseline
from .dqn import DQNBaseline
from .mappo import MAPPOBaseline
from .maddpg import MADDPGBaseline

__all__ = [
    "FinConCouncil",
    "TradingAgentsCouncil",
    "FinMemCouncil",
    "CryptoTradeCouncil",
    "FSReasoningAgentCouncil",
    "A2CBaseline",
    "PPOBaseline",
    "DQNBaseline",
    "MAPPOBaseline",
    "MADDPGBaseline",
]
