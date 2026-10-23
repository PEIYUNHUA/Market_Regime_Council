""""""
from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_agent import BaseAgent, AgentEvaluation
from ..llm.base import BaseLLM
from ..data_layer.base import EnvironmentSnapshot
from ..prompts.templates import ENVIRONMENT_OBSERVER_SYSTEM, ENVIRONMENT_OBSERVER_USER


class EnvironmentObserver(BaseAgent):

    def __init__(self, llm: BaseLLM, system_prompt: str = None):
        super().__init__(
            name="environment_observer (agent 3)", llm=llm,
            system_prompt=system_prompt or ENVIRONMENT_OBSERVER_SYSTEM,
        )

    async def analyze(
        self,
        data: EnvironmentSnapshot,
        fred_df: Optional[pd.DataFrame] = None,
        fng_df: Optional[pd.DataFrame] = None,
        tweets_df: Optional[pd.DataFrame] = None,
        polymarket_df: Optional[pd.DataFrame] = None,
        extra_context: str = "",
    ) -> AgentEvaluation:
        """"""
        fred_text        = self._format_fred(data, fred_df)
        fng_text         = self._format_fng(fng_df, data)
        tweets_text      = self._format_tweets(tweets_df)
        polymarket_text  = self._format_polymarket(polymarket_df)

        user_prompt = ENVIRONMENT_OBSERVER_USER.format(
            fred_summary=fred_text,
            fng_summary=fng_text,
            tweets_summary=tweets_text,
            polymarket_summary=polymarket_text,
        )
        response = await self.llm.chat_text(system_prompt=self.system_prompt, user_prompt=user_prompt)

        parsed = self._parse_json(response)
        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("scores", {}),
            confidence=parsed.get("regime_confidence", parsed.get("confidence", 0.0)),
            key_observations=parsed.get("key_observations", []),
            dominant_dimension=parsed.get("dominant_dimension", "macro"),
        )
        self._history.append(eval_result)
        return eval_result

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _format_fred(self, snap: EnvironmentSnapshot,
                     fred_df: Optional[pd.DataFrame]) -> str:
        """"""

        col_map = {
            "fed_funds_rate":        ("Fed Funds Rate (%)", snap.fed_rate),
            "us_10y_real_yield":     ("US 10Y Real Yield (%)", snap.us10y_yield),
            "vix":                   ("VIX", snap.vix),
            "sp500":                 ("S&P 500", snap.sp500),
            "usd_index_broad":       ("USD Broad Index (DXY proxy)", snap.dxy),
            "wti_oil":               ("WTI Oil ($/bbl)", snap.extra.get("wti_oil")),
            "inflation_expect_5y5y": ("5Y5Y Inflation Expect (%)", snap.extra.get("inflation_expect_5y5y")),
            "fed_balance_sheet":     ("Fed Balance Sheet ($T)", snap.extra.get("fed_balance_sheet_m")),
            "nasdaq100":             ("Nasdaq-100", snap.extra.get("nasdaq100")),
        }

        lines = []
        for col, (label, snap_val) in col_map.items():
            latest_val = snap_val
            chg_str = ""

            if fred_df is not None and col in fred_df.columns and len(fred_df) > 1:
                series = fred_df[col].dropna()
                if len(series) >= 2:
                    v_now = float(series.iloc[-1])
                    v_30d = float(series.iloc[max(0, len(series) - 30)])
                    latest_val = v_now
                    delta = v_now - v_30d
                    chg_str = f" | 30d Δ: {delta:+.4f}"

            if latest_val is not None:
                try:
                    lines.append(f"{label}: {float(latest_val):.4f}{chg_str}")
                except (TypeError, ValueError):
                    pass
            else:
                lines.append(f"{label}: N/A")

        if not lines:
            lines = [
                f"DXY: {snap.dxy or 'N/A'}",
                f"VIX: {snap.vix or 'N/A'}",
                f"S&P500: {snap.sp500 or 'N/A'}",
                f"US10Y Real Yield: {snap.us10y_yield or 'N/A'}%",
                f"Fed Rate: {snap.fed_rate or 'N/A'}%",
            ]
        return "\n".join(lines)

    def _format_fng(self, fng_df: Optional[pd.DataFrame],
                    snap: EnvironmentSnapshot) -> str:
        """"""
        lines = []

        if fng_df is not None and len(fng_df) > 0:

            latest = fng_df.iloc[-1]
            current_val = int(latest.get("fng_classification", snap.fear_greed_index or 0))
            current_label = latest.get("fng_value", self._fng_label(current_val))
            lines.append(f"Current F&G Index: {current_val} ({current_label})")


            recent_28 = fng_df.tail(28)
            avg_28 = float(recent_28["fng_classification"].mean())
            lines.append(f"4-Week Average: {avg_28:.1f}")


            if len(fng_df) >= 14:
                avg_recent_7 = float(fng_df.tail(7)["fng_classification"].mean())
                avg_prev_7 = float(fng_df.tail(14).head(7)["fng_classification"].mean())
                if avg_recent_7 > avg_prev_7 + 3:
                    trend = "improving (increasing greed)"
                elif avg_recent_7 < avg_prev_7 - 3:
                    trend = "deteriorating (increasing fear)"
                else:
                    trend = "stable"
                lines.append(f"Trend (7d vs prior 7d): {trend}")


            weeks_text = []
            for w in range(4, 0, -1):
                wk = fng_df.tail(w * 7).head(7)
                if len(wk) > 0:
                    weeks_text.append(f"Week-{w}: {float(wk['fng_classification'].mean()):.0f}")
            lines.append("Weekly averages (oldest→latest): " + " | ".join(reversed(weeks_text)))

        elif snap.fear_greed_index is not None:
            label = self._fng_label(snap.fear_greed_index)
            lines.append(f"Current F&G Index: {snap.fear_greed_index} ({label})")
            lines.append("Historical data: not available")
        else:
            lines.append("Fear & Greed data not available.")

        return "\n".join(lines)

    @staticmethod
    def _fng_label(val: int) -> str:
        if val <= 20:
            return "Extreme Fear"
        elif val <= 40:
            return "Fear"
        elif val <= 60:
            return "Neutral"
        elif val <= 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def _format_tweets(self, tweets_df: Optional[pd.DataFrame]) -> str:
        """"""
        if tweets_df is None or len(tweets_df) == 0:
            return "Twitter/X data not available for this period (sparse pre-2025)."

        _SENT_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        df = tweets_df.copy()
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df["sent_num"]  = df["sentiment"].map(_SENT_SCORE).fillna(0.0)
        df["engagement"] = (
            df.get("likes",   pd.Series(0, index=df.index)).fillna(0)
            + df.get("reposts", pd.Series(0, index=df.index)).fillna(0)
            + df.get("views",   pd.Series(0, index=df.index)).fillna(0) / 100.0
        ).clip(lower=0).apply(lambda x: math.log1p(x) + 1.0)

        total_n = len(df)
        total_w = df["engagement"].sum()
        overall_wscore = round(
            (df["sent_num"] * df["confidence"] * df["engagement"]).sum() / total_w, 3
        ) if total_w > 0 else 0.0
        counts = df["sentiment"].value_counts()
        bull_pct = round(counts.get("bullish", 0) / total_n * 100, 1)
        bear_pct = round(counts.get("bearish", 0) / total_n * 100, 1)

        lines = [
            f"Total tweets: {total_n} | Overall weighted sentiment: {overall_wscore:+.3f}",
            f"Bullish: {bull_pct}%  Bearish: {bear_pct}%  Neutral: {100-bull_pct-bear_pct:.1f}%",
        ]

        # Per-token breakdown (only tokens with data)
        from ..config import ALL_TOKENS
        token_lines = []
        for sym in ALL_TOKENS:
            mask = df["affected_cryptos"].str.upper().str.split("|").apply(
                lambda toks: sym in [t.strip() for t in toks]
            )
            sub = df[mask]
            if sub.empty:
                continue
            tw = sub["engagement"].sum()
            ws = round(
                (sub["sent_num"] * sub["confidence"] * sub["engagement"]).sum() / tw, 3
            ) if tw > 0 else 0.0
            dom = sub["sentiment"].value_counts().index[0]
            token_lines.append(f"  {sym}: n={len(sub)} sentiment={ws:+.3f} ({dom})")

        if token_lines:
            lines.append("Per-token breakdown:")
            lines.extend(token_lines)
        else:
            lines.append("No per-token tweet data available.")

        return "\n".join(lines)

    def _format_polymarket(self, polymarket_df: Optional[pd.DataFrame]) -> str:
        """"""
        if polymarket_df is None or len(polymarket_df) == 0:
            return "Polymarket data not available."

        from ..config import ALL_TOKENS
        pm_keywords = {
            "BTC":  ["btc", "bitcoin"],
            "ETH":  ["eth", "ethereum", "eip"],
            "BNB":  ["bnb", "binance"],
            "ADA":  ["ada", "cardano"],
            "DOGE": ["doge", "dogecoin"],
            "LINK": ["chainlink", "link"],
            "XRP":  ["xrp", "ripple"],
            "SOL":  ["sol", "solana"],
            "LTC":  ["ltc", "litecoin"],
        }

        if "outcome_norm" not in polymarket_df.columns:
            return "Polymarket data not available (missing outcome_norm column)."
        yes_df = polymarket_df[polymarket_df["outcome_norm"].eq("yes")]
        has_question = "question" in yes_df.columns

        lines = []
        matched_any = False
        for sym in ALL_TOKENS:
            kws = pm_keywords.get(sym, [sym.lower()])
            pattern = "|".join(kws)
            mask = yes_df["slug"].str.contains(pattern, case=False, na=False)
            if has_question:
                mask = mask | yes_df["question"].str.contains(pattern, case=False, na=False)
            sub = yes_df[mask]
            if sub.empty:
                lines.append(f"{sym}: No Polymarket data")
            else:
                matched_any = True
                avg_p = float(sub["price"].mean())
                sentiment = round((avg_p - 0.5) * 2, 3)
                lines.append(f"{sym}: avg_yes_prob={avg_p:.3f} → sentiment_score={sentiment:+.3f} (n={len(sub)})")

        if not matched_any and len(yes_df) > 0:
            n_markets = yes_df["slug"].nunique()
            overall_avg = float(yes_df["price"].mean())
            overall_sent = round((overall_avg - 0.5) * 2, 3)
            sample_slugs = "; ".join(yes_df["slug"].unique()[:3])
            lines.append(
                f"[General Crypto Context] No per-token price markets found. "
                f"{n_markets} available markets (avg_yes_prob={overall_avg:.3f}, "
                f"sentiment={overall_sent:+.3f}). Sample topics: {sample_slugs}"
            )

        return "\n".join(lines)
