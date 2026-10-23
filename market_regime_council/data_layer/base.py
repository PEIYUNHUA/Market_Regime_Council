""""""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KlineBar:
    """"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "1h"


@dataclass
class TokenSnapshot:
    """"""
    symbol: str
    price: float
    price_change_24h: float
    market_cap: float
    volume_24h: float
    klines: Dict[str, List[KlineBar]] = field(default_factory=dict)
    kline_image_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnchainMetrics:
    """"""
    symbol: str
    chain: str
    active_addresses_24h: int = 0
    transaction_count_24h: int = 0
    whale_transactions: List[Dict[str, Any]] = field(default_factory=list)
    dex_volume_24h: float = 0.0
    net_flow_exchange: float = 0.0
    social_sentiment: Optional[Dict[str, Any]] = None
    social_mentions_24h: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentSnapshot:
    """"""
    dxy: Optional[float] = None
    gold_price: Optional[float] = None
    sp500: Optional[float] = None
    us10y_yield: Optional[float] = None
    vix: Optional[float] = None
    fed_rate: Optional[float] = None
    btc_dominance: Optional[float] = None
    total_crypto_mcap: Optional[float] = None
    fear_greed_index: Optional[int] = None
    recent_news: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataBundle:
    """"""
    timestamp: int = field(default_factory=lambda: int(time.time()))
    tokens: Dict[str, TokenSnapshot] = field(default_factory=dict)
    onchain: Dict[str, OnchainMetrics] = field(default_factory=dict)
    environment: Optional[EnvironmentSnapshot] = None


class BaseDataProvider(ABC):
    """"""

    def __init__(self, name: str):
        self.name = name
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}
        self.cache_ttl = 300

    def _get_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if time.time() - self._cache_ts.get(key, 0) < self.cache_ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    @abstractmethod
    async def fetch(self, **kwargs) -> Any:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
