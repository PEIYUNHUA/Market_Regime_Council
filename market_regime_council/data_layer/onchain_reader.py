""""""
from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_ONCHAIN_FILE = os.path.join(_PROJECT_ROOT, "data", "on_chain_data", "onchain_filtered.xlsx")


class OnchainReader:
    """"""

    def __init__(self):
        # {SYM_UPPER: DataFrame(index=date, columns=metrics)}
        self._data: Dict[str, pd.DataFrame] = {}
        try:
            self._data = self._parse(_ONCHAIN_FILE)
            logger.info(f"[OnchainReader] Loaded {len(self._data)} tokens: "
                        f"{sorted(self._data.keys())}")
        except Exception as e:
            logger.warning(f"[OnchainReader] Cannot load onchain_filtered.xlsx: {e}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        return sorted(self._data.keys())

    def get_range(self, symbol: str,
                  start_date: datetime.date,
                  end_date: datetime.date) -> pd.DataFrame:
        """"""
        df = self._data.get(symbol.upper())
        if df is None or df.empty:
            return pd.DataFrame()
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    def ok(self) -> bool:
        return len(self._data) > 0

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def _parse(path: str) -> Dict[str, pd.DataFrame]:
        """"""
        raw        = pd.read_excel(path)
        metric_row = raw.iloc[0].to_dict()
        date_col   = raw.columns[0]


        data = raw.iloc[3:].copy().rename(columns={date_col: "date"})
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).set_index("date")


        all_cols    = [c for c in raw.columns if c != date_col]
        base_tokens = sorted(set(c.split(".")[0] for c in all_cols))

        result: Dict[str, pd.DataFrame] = {}
        for token in base_tokens:
            cols = [c for c in all_cols if c == token or c.startswith(f"{token}.")]
            sub  = data[cols].rename(
                columns={c: str(metric_row.get(c, c)) for c in cols}
            )
            result[token.upper()] = sub.apply(pd.to_numeric, errors="coerce")
        return result
