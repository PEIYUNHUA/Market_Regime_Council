""""""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_FRED_RAW     = os.path.join(_PROJECT_ROOT, "data", "macro_data", "fred_macro_raw.xlsx")
_FNG_CSV      = os.path.join(_PROJECT_ROOT, "data", "macro_data", "fear_greed_index.csv")
_CRYPTO_MACRO = os.path.join(_PROJECT_ROOT, "data", "macro_data", "crypto_macro_raw.xlsx")


class MacroReader:
    """"""

    def __init__(self):
        self.fred_df: Optional[pd.DataFrame] = None
        self.fng_df:  Optional[pd.DataFrame] = None
        self.btc_df:  Optional[pd.DataFrame] = None
        self._load()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def get_fred_range(self, start_date: datetime.date,
                       end_date: datetime.date) -> Optional[pd.DataFrame]:
        """"""
        return self._slice(self.fred_df, start_date, end_date)

    def get_fng_range(self, start_date: datetime.date,
                      end_date: datetime.date) -> Optional[pd.DataFrame]:
        """"""
        return self._slice(self.fng_df, start_date, end_date)

    def get_btc_range(self, start_date: datetime.date,
                      end_date: datetime.date) -> Optional[pd.DataFrame]:
        """"""
        return self._slice(self.btc_df, start_date, end_date)

    def ok(self) -> bool:
        return any(df is not None for df in (self.fred_df, self.fng_df, self.btc_df))

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _load(self):
        # 1. FRED
        try:
            df = pd.read_excel(_FRED_RAW, parse_dates=["date"])
            self.fred_df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"[MacroReader] FRED: {len(self.fred_df)} rows, "
                        f"{self.fred_df['date'].min().date()} → "
                        f"{self.fred_df['date'].max().date()}, "
                        f"cols={list(self.fred_df.columns[1:])}")
        except Exception as e:
            logger.warning(f"[MacroReader] Cannot load fred_macro_raw.xlsx: {e}")

        # 2. Fear & Greed
        try:
            df = pd.read_csv(_FNG_CSV)
            df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
            df["fng_classification"] = pd.to_numeric(df["fng_classification"], errors="coerce")
            self.fng_df = (df.dropna(subset=["date"])
                             .sort_values("date")
                             .reset_index(drop=True))
            logger.info(f"[MacroReader] FNG: {len(self.fng_df)} rows, "
                        f"{self.fng_df['date'].min().date()} → "
                        f"{self.fng_df['date'].max().date()}")
        except Exception as e:
            logger.warning(f"[MacroReader] Cannot load fear_greed_index.csv: {e}")


        try:
            df = pd.read_excel(_CRYPTO_MACRO)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            else:
                df = pd.read_excel(_CRYPTO_MACRO, index_col=0)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df.index.name = "date"
                df = df.reset_index()
            self.btc_df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"[MacroReader] BTC chain: {len(self.btc_df)} rows, "
                        f"{self.btc_df['date'].min().date()} → "
                        f"{self.btc_df['date'].max().date()}, "
                        f"cols={list(self.btc_df.columns[1:])}")
        except Exception as e:
            logger.warning(f"[MacroReader] Cannot load crypto_macro_raw.xlsx: {e}")

    @staticmethod
    def _slice(df: Optional[pd.DataFrame],
               start_date: datetime.date,
               end_date: datetime.date) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        mask  = (df["date"] >= start) & (df["date"] <= end)
        result = df[mask].copy().reset_index(drop=True)
        return result if not result.empty else None
