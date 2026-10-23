""""""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from .base import KlineBar

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_HIST_DIR     = os.path.join(_PROJECT_ROOT, "data", "historical_data")


class HistoricalDataReader:
    """"""

    def __init__(self):
        # {SYM: h5_path}
        self._h5_files: Dict[str, str] = {}
        for fp in glob.glob(os.path.join(_HIST_DIR, "*USDT_*.h5")):
            sym = os.path.basename(fp).split("USDT")[0].upper()
            if sym not in self._h5_files or fp > self._h5_files[sym]:
                self._h5_files[sym] = fp
        logger.info(f"[HistoricalDataReader] {len(self._h5_files)} H5 files: "
                    f"{sorted(self._h5_files.keys())}")


        self._cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        return sorted(self._h5_files.keys())

    def get_bars(self, symbol: str,
                 start_date: "datetime.date",
                 end_date: "datetime.date") -> pd.DataFrame:
        """"""
        df = self._load(symbol.upper())
        if df is None or df.empty:
            return pd.DataFrame()
        mask = (df["Open Time"].dt.date >= start_date) & (df["Open Time"].dt.date <= end_date)
        return df[mask].reset_index(drop=True)

    def get_price(self, symbol: str, date: "datetime.date") -> Optional[float]:
        """"""
        df = self._load(symbol.upper())
        if df is None or df.empty:
            return None
        sub = df[df["Open Time"].dt.date <= date]
        return float(sub["Close"].iloc[-1]) if len(sub) > 0 else None

    def to_kline_bars(self, df: pd.DataFrame, interval: str = "1d") -> List[KlineBar]:
        """"""
        bars = []
        for _, row in df.iterrows():
            ot = row.get("Open Time")
            if ot is None:
                ts = int(time.time())
            elif hasattr(ot, "timestamp"):
                ts = int(ot.timestamp())
            else:
                ts = int(pd.Timestamp(ot).timestamp())
            bars.append(KlineBar(
                timestamp=ts,
                open=float(row.get("Open", 0)),
                high=float(row.get("High", 0)),
                low=float(row.get("Low", 0)),
                close=float(row.get("Close", 0)),
                volume=float(row.get("Volume", 0)),
                interval=interval,
            ))
        return bars

    def ok(self) -> bool:
        return len(self._h5_files) > 0

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _load(self, symbol: str) -> Optional[pd.DataFrame]:
        """"""
        if symbol in self._cache:
            return self._cache[symbol]
        h5 = self._h5_files.get(symbol)
        if not h5:
            return None
        try:
            df = pd.read_hdf(h5, key="stock_data")
            df.columns = [str(c) for c in df.columns]
            if "Open Time" in df.columns:
                df["Open Time"] = pd.to_datetime(df["Open Time"])
            df = df.sort_values("Open Time").reset_index(drop=True)
            self._cache[symbol] = df
            return df
        except Exception as e:
            logger.warning(f"[HistoricalDataReader] Cannot load {symbol}: {e}")
            return None
