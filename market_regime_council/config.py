""""""
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class DominatorType(Enum):
    MARKET = "market"
    TRADE = "trade"
    ENVIRONMENT = "environment"
    OTHER = "other"


ALL_TOKENS = ["ADA", "BCH", "BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "PAXG", "TRX", "XLM", "XRP", "ZEC"]


@dataclass
class LLMConfig:
    """"""

    local_model_path: str = ""
    local_model_name: str = "qwen3-vl"
    local_device: str = "cuda"
    local_dtype: str = "bfloat16"
    local_max_tokens: int = 4096
    local_temperature: float = 0.7
    local_api_base: str = "http://localhost:8000/v1"


    api_provider: str = "openai"
    api_key: str = ""
    api_base_url: str = ""
    api_model_name: str = "gpt-4o"
    api_max_tokens: int = 4096
    api_temperature: float = 0.7


    use_local: bool = False


@dataclass
class DataConfig:
    """"""

    tokens: List[str] = field(default_factory=lambda: list(ALL_TOKENS))

    kline_intervals: List[str] = field(default_factory=lambda: ["1d"])

    lookback_days: int = 90

    kline_style: str = "with_ma"
    chains: List[str] = field(default_factory=lambda: ["ethereum", "bsc", "solana"])
    social_platforms: List[str] = field(default_factory=lambda: ["twitter", "reddit", "telegram"])
    macro_indexes: List[str] = field(default_factory=lambda: [
        "DXY", "GOLD", "SP500", "US10Y", "VIX", "FED_RATE",
    ])
    cache_ttl_seconds: int = 300


@dataclass
class AgentConfig:
    """"""
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "market": 0.25, "trade": 0.25, "environment": 0.25, "other": 0.25,
    })
    reweight_lookback_days: int = 7
    reweight_min_weight: float = 0.05
    reweight_max_weight: float = 0.70
    max_discussion_rounds: int = 3
    confidence_threshold: float = 0.6


@dataclass
class PortfolioConfig:
    """"""
    initial_capital: float = 100000.0
    max_position_pct: float = 1.0
    min_position_pct: float = 0.0
    max_cash_pct: float = 1.0
    transaction_cost_bps: float = 0.0
    allow_short: bool = False
    max_short_weight: float = 0.20


@dataclass
class BacktestConfig:
    """"""
    start_date: datetime.date = field(default_factory=lambda: datetime.date(2021, 1, 1))
    end_date: datetime.date = field(default_factory=lambda: datetime.date(2024, 12, 31))
    rebalance_freq_days: int = 1
    transaction_cost_bps: float = 0.0
    max_weight: float = 1.0
    min_weight: float = 0.0
    max_cash: float = 1.0
    risk_free_rate: float = 0.0
    allow_short: bool = False
    max_short_weight: float = 0.20


@dataclass
class AppConfig:
    """"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


DEFAULT_CONFIG = AppConfig()
