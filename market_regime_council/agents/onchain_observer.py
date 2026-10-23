""""""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_agent import BaseAgent, AgentEvaluation
from ..llm.base import BaseLLM
from ..data_layer.base import OnchainMetrics
from ..prompts.templates import ONCHAIN_OBSERVER_SYSTEM, ONCHAIN_OBSERVER_USER


class OnchainObserver(BaseAgent):

    def __init__(self, llm: BaseLLM, system_prompt: str = None):
        super().__init__(
            name="onchain_observer (agent 2)", llm=llm,
            system_prompt=system_prompt or ONCHAIN_OBSERVER_SYSTEM,
        )

    async def analyze(
        self,
        data: Dict[str, OnchainMetrics],
        btc_macro_df: Optional[pd.DataFrame] = None,
        extra_context: str = "",
    ) -> AgentEvaluation:
        """"""
        onchain_table  = self._format_onchain_table(data)

        zscore_section = self._format_zscore_summary(data)
        if zscore_section:
            onchain_table = onchain_table + "\n\n" + zscore_section
        btc_macro_text = self._format_btc_macro(data, btc_macro_df)

        user_prompt = ONCHAIN_OBSERVER_USER.format(
            onchain_metrics_table=onchain_table,
            btc_macro_summary=btc_macro_text,
        )
        response = await self.llm.chat_text(system_prompt=self.system_prompt, user_prompt=user_prompt)

        parsed = self._parse_json(response)
        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("scores", {}),
            confidence=parsed.get("regime_confidence", parsed.get("confidence", 0.0)),
            key_observations=parsed.get("key_observations", []),
            dominant_dimension=parsed.get("dominant_dimension", "onchain"),
        )
        self._history.append(eval_result)
        return eval_result

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _format_onchain_table(self, data: Dict[str, OnchainMetrics]) -> str:
        """"""

        _ORDER = ["AdrActCnt", "TxCnt", "TokenTxCnt", "FeeTotNtv", "ChainTVL_USD"]
        all_cols: list = []
        for m in data.values():
            for col in m.extra.get("raw_metrics", {}).keys():
                if col not in all_cols:
                    all_cols.append(col)

        ordered = [c for c in _ORDER if c in all_cols] + \
                  [c for c in all_cols if c not in _ORDER]

        if not ordered:

            header = f"{'Token':<6} | {'ActiveAddr':>12} | {'TxCount':>10} | {'DEXVol_USD':>14}"
            sep = "-" * len(header)
            rows = [header, sep]
            for sym, m in sorted(data.items()):
                rows.append(
                    f"{sym:<6} | {m.active_addresses_24h:>12,} | "
                    f"{m.transaction_count_24h:>10,} | {m.dex_volume_24h:>14,.0f}"
                )
            return "\n".join(rows)

        col_w = 14
        header = f"{'Token':<6} | " + " | ".join(f"{c:>{col_w}}" for c in ordered)
        sep = "-" * len(header)
        rows = [header, sep]
        for sym, m in sorted(data.items()):
            raw = m.extra.get("raw_metrics", {})
            cells = []
            for col in ordered:
                if col in raw:
                    v = raw[col]

                    if abs(v) >= 1:
                        cells.append(f"{v:>{col_w},.0f}")
                    else:
                        cells.append(f"{v:>{col_w}.4f}")
                else:
                    cells.append(f"{'N/A':>{col_w}}")
            rows.append(f"{sym:<6} | " + " | ".join(cells))
        return "\n".join(rows)

    def _format_zscore_summary(self, data: Dict[str, OnchainMetrics]) -> str:
        """"""
        lines = []
        threshold = 0.5
        strong_threshold = 1.5

        for sym in sorted(data.keys()):
            m = data[sym]
            z_scores = m.extra.get("z_scores", {})
            if not z_scores:
                continue

            significant = [(col, z) for col, z in z_scores.items() if abs(z) >= threshold]
            if not significant:
                continue

            significant.sort(key=lambda x: -abs(x[1]))
            parts = []
            has_strong = False
            for col, z in significant:
                arrow = "↑" if z > 0 else "↓"
                part = f"{col} {z:+.1f}σ{arrow}"
                parts.append(part)
                if abs(z) >= strong_threshold:
                    has_strong = True

            flag = "  [STRONG SIGNAL]" if has_strong else ""
            lines.append(f"  {sym:<6}: {('  '.join(parts))}{flag}")

        if not lines:
            return ""

        header = "--- 30-Day Z-Score Signal Summary (deviation from 30d mean, |z|≥0.5 shown) ---"
        return header + "\n" + "\n".join(lines)

    def _format_btc_macro(self, data: Dict[str, OnchainMetrics],
                          btc_macro_df: Optional[pd.DataFrame]) -> str:
        """"""
        lines = []


        if btc_macro_df is not None and len(btc_macro_df) > 0:
            latest = btc_macro_df.iloc[-1]
            prev30 = btc_macro_df.iloc[max(0, len(btc_macro_df) - 30)]

            def _val(col: str, row) -> Optional[float]:
                v = row.get(col) if hasattr(row, "get") else row[col] if col in btc_macro_df.columns else None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            def _chg(col: str) -> str:
                v = _val(col, latest)
                p = _val(col, prev30)
                if v is None:
                    return "N/A"
                if p is None or p == 0:
                    return f"{v:.4g}"
                pct = (v / p - 1) * 100
                return f"{v:.4g} ({pct:+.1f}% vs 30d ago)"

            lines.append(f"Hash Rate: {_chg('hash-rate')} EH/s")
            lines.append(f"Realized Vol 30d: {_chg('realized_vol_30d')}")
            lines.append(f"Funding Rate: {_chg('funding_rate')}")
            lines.append(f"Basis: {_chg('basis')}")
            lines.append(f"Active Addresses: {_chg('n-unique-addresses')}")
            lines.append(f"TX Volume USD: {_chg('estimated-transaction-volume-usd')}")


        elif "BTC" in data:
            ex = data["BTC"].extra
            lines.append(f"Hash Rate: {ex.get('hash_rate', 'N/A')}")
            lines.append(f"Realized Vol 30d: {ex.get('realized_vol_30d', 'N/A')}")
            lines.append(f"Funding Rate: {ex.get('funding_rate', 'N/A')}")
            lines.append(f"Basis: {ex.get('basis', 'N/A')}")
        else:
            lines.append("BTC macro data not available.")

        return "\n".join(lines)


