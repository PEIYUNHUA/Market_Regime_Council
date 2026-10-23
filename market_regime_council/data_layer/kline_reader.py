""""""
from __future__ import annotations

import datetime
import glob
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_KLINE_DIR    = os.path.join(_PROJECT_ROOT, "data", "kline_data")


class KlineReader:
    """"""

    def __init__(self, style: str = "with_ma"):
        if style not in ("with_ma", "without_ma"):
            raise ValueError(f"style must be 'with_ma' or 'without_ma', got {style!r}")
        self.style = style

        # {SYM: sorted_png_list}
        self._pngs: Dict[str, List[str]] = {}
        base = os.path.join(_KLINE_DIR, style)
        if os.path.isdir(base):
            for entry in os.listdir(base):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    sym = entry.upper()
                    self._pngs[sym] = sorted(glob.glob(os.path.join(full, "*.png")))
        logger.info(f"[KlineReader] style={style!r}, "
                    f"{len(self._pngs)} symbol dirs: {sorted(self._pngs.keys())}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        return sorted(self._pngs.keys())

    def get_image(self, symbol: str,
                  target_date: Optional[datetime.date] = None) -> Optional[str]:
        """"""
        pngs = self._pngs.get(symbol.upper(), [])
        if not pngs:
            return None
        if target_date is None:
            return pngs[-1]
        tgt = target_date.strftime("%Y%m%d")
        candidates = [p for p in pngs if self._date_str(p) <= tgt]
        return candidates[-1] if candidates else None

    def get_images_in_range(self, symbol: str,
                             start_date: datetime.date,
                             end_date: datetime.date) -> List[str]:
        """"""
        pngs = self._pngs.get(symbol.upper(), [])
        start_str = start_date.strftime("%Y%m%d")
        end_str   = end_date.strftime("%Y%m%d")
        return [p for p in pngs if start_str <= self._date_str(p) <= end_str]

    def ok(self) -> bool:
        return len(self._pngs) > 0

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def _date_str(png_path: str) -> str:
        """"""
        name  = os.path.splitext(os.path.basename(png_path))[0]
        parts = name.rsplit("_", 1)
        return parts[-1] if len(parts) == 2 and parts[-1].isdigit() else ""
