""""""
from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentEvaluation
from ..llm.base import BaseLLM
from ..data_layer.base import KlineBar, TokenSnapshot
from ..prompts.templates import (
    MARKET_OBSERVER_SYSTEM, MARKET_OBSERVER_USER, MARKET_OBSERVER_MULTIMODAL_USER,
)


class MarketObserver(BaseAgent):

    def __init__(self, llm: BaseLLM, system_prompt: str = None):
        super().__init__(
            name="market_observer (agent 1)", llm=llm,
            system_prompt=system_prompt or MARKET_OBSERVER_SYSTEM,
        )

    async def analyze(
        self,
        data: Dict[str, TokenSnapshot],
        kline_images: Dict[str, str] = None,
        lookback_days: int = 90,
        extra_context: str = "",
    ) -> AgentEvaluation:
        """"""
        stats_table = self._format_stats_table(data, lookback_days)
        kline_text = self._format_kline_summary(data)

        if kline_images and hasattr(self.llm, 'chat_multimodal'):
            image_paths = [v for v in kline_images.values() if v]
            if image_paths:
                response = await self.llm.chat_multimodal(
                    system_prompt=self.system_prompt,
                    text=MARKET_OBSERVER_MULTIMODAL_USER.format(
                        lookback_days=lookback_days,
                        token_stats_table=stats_table,
                    ),
                    image_paths=image_paths,
                )
            else:
                response = await self._call_text(stats_table, kline_text, lookback_days, extra_context)
        else:
            response = await self._call_text(stats_table, kline_text, lookback_days, extra_context)

        parsed = self._parse_json(response)
        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("scores", {}),
            confidence=parsed.get("regime_confidence", parsed.get("confidence", 0.0)),
            key_observations=parsed.get("key_observations", []),
            dominant_dimension=parsed.get("dominant_dimension", "technical"),
        )
        self._history.append(eval_result)
        return eval_result

    async def _call_text(self, stats_table: str, kline_text: str,
                         lookback_days: int, extra_context: str) -> str:
        user_prompt = MARKET_OBSERVER_USER.format(
            lookback_days=lookback_days,
            token_stats_table=stats_table,
            kline_summary=kline_text,
            extra_context=extra_context,
        )
        return await self.llm.chat_text(system_prompt=self.system_prompt, user_prompt=user_prompt)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _format_stats_table(self, data: Dict[str, TokenSnapshot], lookback_days: int) -> str:
        """"""
        header = (
            f"{'Token':<6} | {'Close':>10} | {'1d%':>7} | {'7d%':>7} | {'30d%':>8} | "
            f"{'LB%':>8} | {'30dVol%':>8} | {'VolZ':>6}"
        )
        sep = "-" * len(header)
        rows = [header, sep]

        for sym, snap in sorted(data.items()):
            bars = snap.klines.get("1d", [])
            if not bars:
                rows.append(f"{sym:<6} | {'N/A':>10} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>6}")
                continue

            closes = [b.close for b in bars]
            volumes = [b.volume for b in bars]
            latest = closes[-1]

            ret_1d  = self._return(closes, 1)
            ret_7d  = self._return(closes, 7)
            ret_30d = self._return(closes, 30)
            ret_lb  = self._return(closes, min(lookback_days, len(closes) - 1))
            vol_30d = self._vol(closes, 30) * 100       # annualized in %

            # Volume z-score (recent 7d vs 30d baseline)
            vol_z = self._vol_zscore(volumes, recent=7, baseline=30)

            rows.append(
                f"{sym:<6} | {latest:>10.4f} | {ret_1d:>+7.2f} | {ret_7d:>+7.2f} | "
                f"{ret_30d:>+8.2f} | {ret_lb:>+8.2f} | {vol_30d:>8.1f} | {vol_z:>+6.2f}"
            )

        return "\n".join(rows)

    def _format_kline_summary(self, data: Dict[str, TokenSnapshot]) -> str:
        """"""
        lines = []
        for sym, snap in sorted(data.items()):
            bars = snap.klines.get("1d", [])
            if len(bars) < 5:
                continue
            last_5 = bars[-5:]
            closes = [b.close for b in last_5]
            vols = [b.volume for b in last_5]
            trend = "uptrend" if closes[-1] > closes[0] else "downtrend" if closes[-1] < closes[0] else "flat"
            green = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            lines.append(
                f"[{sym}] Last 5 closes: {[round(c, 4) for c in closes]} | "
                f"Trend: {trend} | Green bars: {green}/4 | Avg vol: {sum(vols)/len(vols):.1f}"
            )
        return "\n".join(lines) if lines else "No kline data available."

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def _return(closes: List[float], n: int) -> float:
        """"""
        if len(closes) <= n or closes[-n - 1] == 0:
            return 0.0
        return round((closes[-1] / closes[-n - 1] - 1) * 100, 2)

    @staticmethod
    def _vol(closes: List[float], n: int) -> float:
        """"""
        if len(closes) < n + 1:
            return 0.0
        rets = [(closes[i] / closes[i - 1] - 1) for i in range(max(1, len(closes) - n), len(closes))]
        if len(rets) < 2:
            return 0.0
        mu = sum(rets) / len(rets)
        variance = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(variance * 252)

    @staticmethod
    def _vol_zscore(volumes: List[float], recent: int = 7, baseline: int = 30) -> float:
        """"""
        if len(volumes) < baseline:
            return 0.0
        baseline_vols = volumes[-baseline:]
        recent_vols = volumes[-recent:]
        mu = sum(baseline_vols) / len(baseline_vols)
        if mu == 0:
            return 0.0
        sigma = math.sqrt(sum((v - mu) ** 2 for v in baseline_vols) / len(baseline_vols))
        if sigma == 0:
            return 0.0
        recent_mu = sum(recent_vols) / len(recent_vols)
        return round((recent_mu - mu) / sigma, 2)
