""""""
from __future__ import annotations

import datetime
import logging
import math
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_POLYMARKET   = os.path.join(_PROJECT_ROOT, "data", "social_media_data",
                              "polymarket_crypto_dump", "crypto_prices_filtered.csv")
_TWEETS       = os.path.join(_PROJECT_ROOT, "data", "social_media_data",
                              "Tweets", "tweets_all.csv")


_KEYWORDS: Dict[str, List[str]] = {
    "BTC":  ["btc", "bitcoin"],
    "ETH":  ["eth", "ethereum"],
    "BNB":  ["bnb", "binance"],
    "ADA":  ["ada", "cardano"],
    "DOGE": ["doge", "dogecoin"],
    "LINK": ["chainlink", "link"],
    "SOL":  ["sol", "solana"],
    "LTC":  ["ltc", "litecoin"],
    "XRP":  ["xrp", "ripple"],
}


class SocialMediaReader:
    """"""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        try:
            df = pd.read_csv(_POLYMARKET)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            self._df = (df.dropna(subset=["date"])
                          .sort_values("date")
                          .reset_index(drop=True))
            logger.info(f"[SocialMediaReader] Polymarket: {len(self._df)} rows, "
                        f"{self._df['date'].min().date()} → {self._df['date'].max().date()}")
        except Exception as e:
            logger.warning(f"[SocialMediaReader] Cannot load Polymarket CSV: {e}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def get_range(self, start_date: datetime.date,
                  end_date: datetime.date) -> pd.DataFrame:
        """"""
        if self._df is None:
            return pd.DataFrame()
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        mask  = (self._df["date"] >= start) & (self._df["date"] <= end)
        return self._df[mask].copy().reset_index(drop=True)

    def get_sentiment(self, symbol: str,
                      start_date: datetime.date,
                      end_date: datetime.date) -> Dict:
        """"""
        df = self.get_range(start_date, end_date)
        if df.empty:
            return {"symbol": symbol, "overall_sentiment": 0.0,
                    "mentions": 0, "source": "no_data"}

        keywords = _KEYWORDS.get(symbol.upper(), [symbol.lower()])
        pattern  = "|".join(keywords)
        mask     = df["slug"].str.contains(pattern, case=False, na=False)
        yes_rows = df[mask & (df["outcome_norm"] == "yes")]

        if yes_rows.empty:
            return {"symbol": symbol, "overall_sentiment": 0.0,
                    "mentions": 0, "source": "no_match"}

        avg_prob = float(yes_rows["price"].mean())
        return {
            "symbol":             symbol,
            "overall_sentiment":  round((avg_prob - 0.5) * 2, 3),
            "mentions":           len(yes_rows),
            "avg_prob":           round(avg_prob, 3),
            "source":             "polymarket",
        }

    def ok(self) -> bool:
        return self._df is not None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


_SENT_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


class TweetReader:
    """"""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

        self._expanded: Optional[pd.DataFrame] = None
        try:
            df = pd.read_csv(_TWEETS, encoding="utf-8-sig")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (df.dropna(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True))

            df["sentiment"] = df["sentiment"].str.lower().str.strip()

            df["sentiment_score"] = df["sentiment"].map(_SENT_SCORE).fillna(0.0)

            df["engagement"] = (
                df["likes"].fillna(0)
                + df["reposts"].fillna(0)
                + df["views"].fillna(0) / 100.0
            ).clip(lower=0).apply(lambda x: math.log1p(x) + 1.0)
            self._df = df

            self._expanded = self._expand_tokens(df)
            logger.info(
                f"[TweetReader] Loaded {len(df)} tweets "
                f"({df['date'].min().date()} → {df['date'].max().date()}), "
                f"expanded={len(self._expanded)} rows"
            )
        except Exception as e:
            logger.warning(f"[TweetReader] Cannot load tweets CSV: {e}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def get_range(self, start_date: datetime.date,
                  end_date: datetime.date) -> pd.DataFrame:
        """"""
        if self._df is None:
            return pd.DataFrame()
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        mask  = (self._df["date"] >= start) & (self._df["date"] <= end)
        return self._df[mask].copy().reset_index(drop=True)

    def get_token_sentiment(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Dict:
        """"""
        empty = {
            "symbol": symbol, "tweet_count": 0,
            "weighted_sentiment": 0.0, "avg_confidence": 0.0,
            "dominant": "neutral", "bullish_pct": 0.0, "bearish_pct": 0.0,
        }
        if self._expanded is None:
            return empty

        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        mask  = (
            (self._expanded["date"] >= start)
            & (self._expanded["date"] <= end)
            & (self._expanded["token"] == symbol.upper())
        )
        sub = self._expanded[mask]
        if sub.empty:
            return empty

        total_w = sub["engagement"].sum()
        w_sent  = (sub["sentiment_score"] * sub["confidence"] * sub["engagement"]).sum()
        weighted_sentiment = round(w_sent / total_w, 4) if total_w > 0 else 0.0
        avg_conf = round(float(sub["confidence"].mean()), 3)

        n = len(sub)
        counts   = sub["sentiment"].value_counts()
        bullish_pct = round(counts.get("bullish", 0) / n, 3)
        bearish_pct = round(counts.get("bearish", 0) / n, 3)
        dominant    = counts.index[0] if not counts.empty else "neutral"

        return {
            "symbol":               symbol,
            "tweet_count":          n,
            "weighted_sentiment":   weighted_sentiment,
            "avg_confidence":       avg_conf,
            "dominant":             dominant,
            "bullish_pct":          bullish_pct,
            "bearish_pct":          bearish_pct,
        }

    def ok(self) -> bool:
        return self._df is not None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def _expand_tokens(df: pd.DataFrame) -> pd.DataFrame:
        """"""
        records = []
        for _, row in df.iterrows():
            tokens = [t.strip().upper() for t in str(row["affected_cryptos"]).split("|")]
            for tok in tokens:
                if tok:
                    rec = row.to_dict()
                    rec["token"] = tok
                    records.append(rec)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).reset_index(drop=True)
