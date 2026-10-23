""""""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import (KlineBar, TokenSnapshot, OnchainMetrics,
                   EnvironmentSnapshot, MarketDataBundle)
from .historical_data_reader import HistoricalDataReader
from .kline_reader import KlineReader
from .social_media_reader import SocialMediaReader, TweetReader
from .onchain_reader import OnchainReader
from .macro_reader import MacroReader
from ..config import DataConfig

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MACRO_INTRO  = os.path.join(_PROJECT_ROOT, "data", "macro_data", "macro_intro.xlsx")


_CHAIN_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "bsc",
    "ADA": "cardano", "SOL": "solana",   "DOGE": "dogecoin",
    "LTC": "litecoin","XRP": "ripple",   "LINK": "ethereum",
    "TRX": "tron",    "XLM": "stellar",  "ZEC": "zcash",
    "PAXG": "ethereum", "BCH": "bitcoin-cash",
}


class DataManager:

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()


        self.historical = HistoricalDataReader()
        self.kline      = KlineReader(style=self.config.kline_style)
        self.social     = SocialMediaReader()
        self.tweets     = TweetReader()
        self.onchain    = OnchainReader()
        self.macro      = MacroReader()

        self._latest_bundle: Optional[MarketDataBundle] = None


        self._clean_all()

    # ==================================================================

    # ==================================================================

    def _load_macro_intro(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(_MACRO_INTRO)
            logger.info(f"[DataManager] macro_intro: {len(df)} indicators loaded")
            return df
        except Exception as e:
            logger.warning(f"[DataManager] Cannot load macro_intro.xlsx: {e}. "
                           "Cleaning will use all available columns.")
            return pd.DataFrame(columns=["source", "short_name", "frequency"])

    def _clean_all(self):
        """"""
        intro = self._load_macro_intro()


        if self.macro.fred_df is not None:
            fred_intro   = intro[intro["source"] == "fred"] if not intro.empty else pd.DataFrame()
            daily_cols   = (fred_intro[fred_intro["frequency"] == "Daily"]["short_name"].tolist()
                            if not fred_intro.empty else [])
            monthly_cols = (fred_intro[fred_intro["frequency"] == "Monthly"]["short_name"].tolist()
                            if not fred_intro.empty else [])
            all_target   = daily_cols + monthly_cols

            df = (self.macro.fred_df[["date"] + [c for c in all_target
                                                   if c in self.macro.fred_df.columns]].copy()
                  if all_target else self.macro.fred_df.copy())

            for col in daily_cols:
                if col in df.columns:
                    df[col] = df[col].ffill(limit=5)
            for col in monthly_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()

            self.macro.fred_df = df
            logger.info(f"[DataManager] FRED cleaned: {df.shape}, "
                        f"daily={daily_cols}, monthly={monthly_cols}")


        if self.macro.btc_df is not None:
            btc_intro = (intro[intro["source"].isin(["blockchain", "binance"])]
                         if not intro.empty else pd.DataFrame())
            btc_cols  = btc_intro["short_name"].tolist() if not btc_intro.empty else []

            df = (self.macro.btc_df[["date"] + [c for c in btc_cols
                                                  if c in self.macro.btc_df.columns]].copy()
                  if btc_cols else self.macro.btc_df.copy())

            for col in [c for c in df.columns if c != "date"]:
                df[col] = df[col].ffill(limit=5)

            self.macro.btc_df = df
            logger.info(f"[DataManager] BTC chain cleaned: {df.shape}")

        # ── 3. Fear & Greed ───────────────────────────────────────────
        if self.macro.fng_df is not None:
            df = self.macro.fng_df.copy()
            df["fng_classification"] = df["fng_classification"].ffill()
            self.macro.fng_df = df
            logger.info(f"[DataManager] FNG cleaned: {df.shape}")


        if self.onchain._data:
            for sym, df in self.onchain._data.items():
                cleaned = df.copy()
                for col in cleaned.columns:
                    cleaned[col] = cleaned[col].ffill(limit=5)
                self.onchain._data[sym] = cleaned
            logger.info(f"[DataManager] Onchain cleaned: {len(self.onchain._data)} tokens")

    # ==================================================================

    # ==================================================================

    def get_agent1_data(self,
                        symbols: List[str],
                        start_date: "datetime.date",
                        end_date: "datetime.date") -> Dict[str, Any]:
        """"""
        market_snapshots: Dict[str, TokenSnapshot] = {}
        kline_images:     Dict[str, str]           = {}
        prices:           Dict[str, float]          = {}

        for sym in symbols:
            bars_df = self.historical.get_bars(sym, start_date, end_date)

            if not bars_df.empty:
                bars  = self.historical.to_kline_bars(bars_df, "1d")
                last  = bars_df.iloc[-1]
                prev  = bars_df.iloc[-2] if len(bars_df) >= 2 else bars_df.iloc[-1]
                close = float(last.get("Close", 0))
                prev_close = float(prev.get("Close", 0))
                chg   = (close / prev_close - 1) * 100 if prev_close > 0 else 0.0
                vol   = float(last.get("Quote Asset Volume", last.get("Volume", 0)))

                snap = TokenSnapshot(
                    symbol=sym,
                    price=round(close, 6),
                    price_change_24h=round(chg, 4),
                    market_cap=0.0,
                    volume_24h=round(vol, 2),
                    klines={"1d": bars},
                )
                prices[sym] = close
            else:
                snap = self._mock_token(sym)
                prices[sym] = snap.price

            market_snapshots[sym] = snap


            img = self.kline.get_image(sym, target_date=end_date)
            if img:
                kline_images[sym] = img

        return {
            "market_snapshots": market_snapshots,
            "kline_images":     kline_images,
            "prices":           prices,
        }

    def get_agent2_data(self,
                        symbols: List[str],
                        start_date: "datetime.date",
                        end_date: "datetime.date") -> Dict[str, Any]:
        """"""
        onchain_metrics: Dict[str, OnchainMetrics] = {}

        for sym in symbols:
            chain = _CHAIN_MAP.get(sym.upper(), "ethereum")
            df    = self.onchain.get_range(sym, start_date, end_date)

            if not df.empty:
                last = df.iloc[-1]

                def _safe(col, _last=last, _df=df):
                    v = _last[col] if col in _df.columns else None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None


                raw_metrics = {
                    col: round(float(last[col]), 4)
                    for col in df.columns
                    if col in last.index and pd.notna(last[col])
                }


                # z = (current - mean_30d) / std_30d

                z_scores: Dict[str, float] = {}
                if len(df) >= 5:
                    window_df = df.iloc[-30:] if len(df) >= 30 else df
                    for col in raw_metrics:
                        try:
                            vals = window_df[col].dropna()
                            if len(vals) >= 5:
                                mean = float(vals.mean())
                                std  = float(vals.std())
                                if std > 0:
                                    z = (float(last[col]) - mean) / std
                                    z_scores[col] = round(z, 2)
                        except Exception:
                            pass

                metrics = OnchainMetrics(
                    symbol=sym,
                    chain=chain,
                    active_addresses_24h=int(_safe("AdrActCnt") or 0),
                    transaction_count_24h=int(_safe("TxCnt") or 0),
                    dex_volume_24h=_safe("ChainTVL_USD") or 0.0,
                    net_flow_exchange=0.0,
                    extra={
                        "raw_metrics": raw_metrics,
                        "z_scores":    z_scores,
                        "data_date": str(df.index[-1].date()),
                        "source": "onchain_filtered",
                    },
                )
            else:
                metrics = self._mock_onchain(sym, chain, {})

            onchain_metrics[sym] = metrics

        btc_macro_df = self.macro.get_btc_range(start_date, end_date)

        return {
            "onchain_metrics": onchain_metrics,
            "btc_macro_df":    btc_macro_df,
        }

    def get_agent3_data(self,
                        start_date: "datetime.date",
                        end_date: "datetime.date") -> Dict[str, Any]:
        """"""
        fred_df       = self.macro.get_fred_range(start_date, end_date)
        fng_df        = self.macro.get_fng_range(start_date, end_date)
        tweets_df     = self.tweets.get_range(start_date, end_date)
        polymarket_df = self.social.get_range(start_date, end_date)


        env_snapshot = self._build_env_snapshot(fred_df, fng_df)

        return {
            "env_snapshot":  env_snapshot,
            "fred_df":       fred_df,
            "fng_df":        fng_df,
            "tweets_df":     tweets_df     if not tweets_df.empty     else None,
            "polymarket_df": polymarket_df if not polymarket_df.empty else None,
        }

    # ==================================================================

    # ==================================================================

    async def get_data_for_market_observer(self, symbols: List[str] = None
                                           ) -> Dict[str, TokenSnapshot]:
        symbols = symbols or self.config.tokens
        import datetime as _dt
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.config.lookback_days)
        return self.get_agent1_data(symbols, start, end)["market_snapshots"]

    async def get_data_for_onchain_observer(self, symbols: List[str] = None
                                            ) -> Dict[str, OnchainMetrics]:
        symbols = symbols or self.config.tokens
        import datetime as _dt
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.config.lookback_days)
        return self.get_agent2_data(symbols, start, end)["onchain_metrics"]

    async def get_data_for_environment_observer(self) -> EnvironmentSnapshot:
        import datetime as _dt
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.config.lookback_days)
        return self.get_agent3_data(start, end)["env_snapshot"]

    async def get_kline_image(self, symbol: str, interval: str = "1d",
                              target_date=None) -> Optional[str]:
        return self.kline.get_image(symbol, target_date=target_date)

    async def fetch_all(self, symbols: List[str] = None) -> MarketDataBundle:
        symbols = symbols or self.config.tokens
        import datetime as _dt
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.config.lookback_days)
        a1 = self.get_agent1_data(symbols, start, end)
        a2 = self.get_agent2_data(symbols, start, end)
        a3 = self.get_agent3_data(start, end)
        bundle = MarketDataBundle(
            timestamp=int(time.time()),
            tokens=a1["market_snapshots"],
            onchain=a2["onchain_metrics"],
            environment=a3["env_snapshot"],
        )
        self._latest_bundle = bundle
        return bundle

    @property
    def latest_bundle(self) -> Optional[MarketDataBundle]:
        return self._latest_bundle

    async def health_check(self) -> Dict[str, bool]:
        return {
            "historical": self.historical.ok(),
            "kline":      self.kline.ok(),
            "social":     self.social.ok(),
            "tweets":     self.tweets.ok(),
            "onchain":    self.onchain.ok(),
            "macro":      self.macro.ok(),
        }

    # ==================================================================

    # ==================================================================

    @staticmethod
    def _build_env_snapshot(fred_df: Optional[pd.DataFrame],
                            fng_df: Optional[pd.DataFrame]) -> EnvironmentSnapshot:
        """"""

        def _latest(df: Optional[pd.DataFrame], col: str) -> Optional[float]:
            if df is None or col not in df.columns:
                return None
            s = df[col].dropna()
            return float(s.iloc[-1]) if not s.empty else None

        dxy    = _latest(fred_df, "usd_index_broad")
        vix    = _latest(fred_df, "vix")
        us10y  = _latest(fred_df, "us_10y_real_yield")
        wti    = _latest(fred_df, "wti_oil")
        on_rrp = _latest(fred_df, "on_rrp")
        cpi    = _latest(fred_df, "cpi")
        core_pce  = _latest(fred_df, "core_pce")
        unemp     = _latest(fred_df, "unemployment_rate")
        nonfarm   = _latest(fred_df, "nonfarm_payrolls")
        indprod   = _latest(fred_df, "industrial_production")

        fear_greed = None
        fng_date   = None
        if fng_df is not None and len(fng_df) > 0:
            last_fng   = fng_df.iloc[-1]
            fear_greed = int(last_fng["fng_classification"])
            fng_date   = str(last_fng["date"].date())

        return EnvironmentSnapshot(
            dxy=round(dxy, 4)   if dxy   is not None else None,
            gold_price=None,
            sp500=None,
            us10y_yield=round(us10y, 4) if us10y is not None else None,
            vix=round(vix, 2)   if vix   is not None else None,
            fed_rate=None,
            btc_dominance=None,                  # TODO
            total_crypto_mcap=None,              # TODO
            fear_greed_index=fear_greed,
            recent_news=[],
            extra={
                "wti_oil":              wti,
                "on_rrp":               on_rrp,
                "cpi":                  cpi,
                "core_pce":             core_pce,
                "unemployment_rate":    unemp,
                "nonfarm_payrolls":     nonfarm,
                "industrial_production": indprod,
                "fng_date":             fng_date,
            },
        )

    # ==================================================================

    # ==================================================================

    @staticmethod
    def _mock_token(symbol: str) -> TokenSnapshot:
        base_prices = {"BTC": 87000, "ETH": 3200, "SOL": 180, "BNB": 620,
                       "ADA": 0.9, "DOGE": 0.35, "LINK": 15.0}
        base  = base_prices.get(symbol, 1.0)
        price = base * (1 + random.uniform(-0.03, 0.03))
        return TokenSnapshot(
            symbol=symbol,
            price=round(price, 6),
            price_change_24h=round(random.uniform(-5, 5), 2),
            market_cap=0.0,
            volume_24h=round(random.uniform(1e7, 1e9), 0),
            klines={},
        )

    @staticmethod
    def _mock_onchain(symbol: str, chain: str, sentiment: Dict) -> OnchainMetrics:
        return OnchainMetrics(
            symbol=symbol, chain=chain,
            active_addresses_24h=random.randint(10_000, 500_000),
            transaction_count_24h=random.randint(50_000, 2_000_000),
            dex_volume_24h=round(random.uniform(1e7, 1e9), 0),
            net_flow_exchange=0.0,
            social_sentiment=sentiment,
            social_mentions_24h=sentiment.get("mentions", 0),
        )
