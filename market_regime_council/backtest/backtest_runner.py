""""""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import BacktestConfig, ALL_TOKENS
from ..council.regime_council import RegimeCouncil, DateRangeBundle, MarketRegimeDetector
from ..data_layer.data_manager import DataManager
from ..portfolio.portfolio_manager import PortfolioManager
from .metrics import compute_all_metrics
from .result import BacktestResult

logger = logging.getLogger(__name__)


class BacktestRunner:
    """"""

    def __init__(
        self,
        council: RegimeCouncil,
        config: BacktestConfig,
        data_manager: DataManager,
        lookback_days: int = 90,
        kline_style: str = "with_ma",
        baseline_councils: Dict[str, Any] = None,
        run_main_council: bool = True,
        smoothing_alpha: float = 0.70,
        mom_strength: Optional[float] = None,
        incremental_json_path: Optional[str] = None,
        stages_log_path: Optional[str] = None,
    ):
        self.council = council
        self.config = config
        self.data_manager = data_manager
        self.lookback_days = lookback_days
        self.kline_style = kline_style
        self.symbols = ALL_TOKENS
        self._freq_days: int = config.rebalance_freq_days


        self.council.lookback_days = lookback_days


        self.portfolio = PortfolioManager(
            initial_capital=100_000.0,
            transaction_cost_bps=config.transaction_cost_bps,
            max_weight=config.max_weight,
            max_cash_weight=config.max_cash,
            allow_short=config.allow_short,
            max_short_weight=config.max_short_weight,
            name="MRC",
        )


        self.run_main_council: bool = run_main_council


        # new_w = alpha * llm_w + (1-alpha) * prev_w

        self._smoothing_alpha: float = smoothing_alpha


        self._mom_strength: Optional[float] = mom_strength
        self._prev_mrc_weights: Dict[str, float] = {}
        self._prev_mrc_cash: float = 0.0


        # dd_scale = 1.0 - tanh(current_dd / ref) * max_reduction

        self._peak_portfolio_value: float = 0.0


        self._incremental_json_path: Optional[str] = incremental_json_path


        self._stages_log_path: Optional[str] = stages_log_path


        self._regime_detector = MarketRegimeDetector()


        self._prev_regime: Optional[str] = None
        self._transition_boost_agent: Optional[str] = None
        self._transition_remaining: int = 0


        self._smoothed_regime_score: float = 0.0
        self._prev_smoothed_score:  float = 0.0
        self._regime_ema_alpha: float = 0.10  # half-life ≈ 6.6 days

        # LLM-based baselines (FinCon, TradingAgents, FinMem, ...)
        self.baseline_councils: Dict[str, Any] = baseline_councils or {}
        self.baseline_portfolios: Dict[str, PortfolioManager] = {
            name: PortfolioManager(
                initial_capital=100_000.0,
                transaction_cost_bps=config.transaction_cost_bps,
                max_weight=config.max_weight,
                max_cash_weight=config.max_cash,

                name=name.upper(),
            )
            for name in self.baseline_councils
        }
        self._baseline_returns:   Dict[str, List[float]] = {n: [] for n in self.baseline_councils}
        self._baseline_turnovers: Dict[str, List[float]] = {n: [] for n in self.baseline_councils}
        self._baseline_prev_vals: Dict[str, Optional[float]] = {n: None for n in self.baseline_councils}

    def run(self) -> BacktestResult:
        """"""
        return asyncio.run(self.run_async())

    async def run_async(self) -> BacktestResult:
        """"""
        eval_dates = self._rebalance_dates(self.config.start_date, self.config.end_date,
                                           self._freq_days)

        logger.info(
            f"[Backtest] Starting: {self.config.start_date} → {self.config.end_date} | "
            f"{len(eval_dates)} rebalance periods (every {self._freq_days}d) | lookback={self.lookback_days}d"
        )

        # ── Hyperparameter log block ────────────────────────────────────────
        _llm = getattr(self.council, "llm", None)
        _llm_cls  = type(_llm).__name__ if _llm is not None else "N/A"
        _llm_model = getattr(_llm, "model_name", "N/A") if _llm is not None else "N/A"
        _llm_temp  = getattr(_llm, "temperature", "N/A") if _llm is not None else "N/A"
        _llm_ctx   = getattr(_llm, "num_ctx", "N/A")    if _llm is not None else "N/A"
        _llm_pred  = getattr(_llm, "num_predict", "N/A") if _llm is not None else "N/A"
        _bl_names  = list(self.baseline_councils.keys()) if self.baseline_councils else []
        logger.info(
            "[Backtest] Hyperparameters:\n"
            "  LLM backend   : %s\n"
            "  LLM model     : %s\n"
            "  temperature   : %s\n"
            "  num_ctx       : %s  num_predict: %s\n"
            "  run_main_council: %s\n"
            "  baselines     : %s\n"
            "  allow_short   : %s\n"
            "  smoothing_alpha: %.2f\n"
            "  mom_strength  : %s\n"
            "  rebalance_freq: %dd\n"
            "  lookback_days : %d\n"
            "  initial_capital: %.0f\n"
            "  transaction_cost_bps: %.1f\n"
            "  max_weight    : %.2f  max_cash: %.2f\n"
            "  kline_style   : %s",
            _llm_cls, _llm_model, _llm_temp, _llm_ctx, _llm_pred,
            self.run_main_council,
            _bl_names if _bl_names else "(none)",
            self.config.allow_short,
            self._smoothing_alpha,
            self._mom_strength if self._mom_strength is not None else "auto",
            self._freq_days,
            self.lookback_days,
            100_000.0,
            self.config.transaction_cost_bps,
            self.config.max_weight, self.config.max_cash,
            self.kline_style,
        )
        # ───────────────────────────────────────────────────────────────────

        period_returns: List[float] = []
        turnovers: List[float] = []
        shapley_history: List[Dict[str, Any]] = []
        prev_value: Optional[float] = None


        _STAGE_KEYS = ["agent1", "agent2", "agent3", "pair_12", "pair_13", "pair_23", "stage3"]
        _stage_cum: Dict[str, float] = {k: 1.0 for k in _STAGE_KEYS}
        _stages_records: List[Dict[str, Any]] = []
        _prev_prices: Dict[str, float] = {}


        _inc_file = None
        _inc_first = True
        if self._incremental_json_path:
            os.makedirs(os.path.dirname(self._incremental_json_path), exist_ok=True)
            _inc_file = open(self._incremental_json_path, "w", encoding="utf-8")
            _inc_file.write("[\n")
            _inc_file.flush()

        for i, eval_date in enumerate(eval_dates):
            logger.info(f"[Backtest] Period {i+1}/{len(eval_dates)}: {eval_date}")


            _btc_dom: float = 0.0


            lookback_start = eval_date - datetime.timedelta(days=self.lookback_days)
            bundle = await self._fetch_date_range(lookback_start, eval_date)


            _, raw_regime_score = self._detect_market_regime(eval_date, bundle.prices)


            self._prev_smoothed_score = self._smoothed_regime_score
            self._smoothed_regime_score = (
                (1.0 - self._regime_ema_alpha) * self._smoothed_regime_score
                + self._regime_ema_alpha * raw_regime_score
            )
            regime_score = self._smoothed_regime_score

            if regime_score > 0.30:
                regime_str = "bull"
            elif regime_score < -0.30:
                regime_str = "bear"
            else:
                regime_str = "volatile"

            bundle.market_regime           = regime_str
            bundle.regime_score            = regime_score

            _bull_to_volatile = (self._prev_regime == "bull" and regime_str == "volatile")
            bundle.regime_transition_boost = self._compute_transition_boost(regime_str)


            result = {}
            if self.run_main_council:
                try:
                    result = await self.council.run_with_bundle(bundle)
                    weights = result.get("portfolio_weights", {})
                    cash_weight = result.get("cash_weight", 0.0)

                    if "shapley_snapshot" in result:
                        snap = dict(result["shapley_snapshot"])
                        snap["eval_date"] = str(eval_date)
                        snap["agent_weights_used"] = result.get("agent_weights_used", {})
                        snap["pair_weights_used"] = result.get("pair_weights_used", {})
                        snap["market_regime"] = result.get("market_regime", "unknown")
                        snap["regime_multipliers"] = result.get("regime_multipliers_used", {})
                        shapley_history.append(snap)
                except Exception as e:
                    logger.warning(
                        "[Backtest] Council failed on %s (%s: %s). Using equal weight.",
                        eval_date, type(e).__name__, e,
                    )
                    n = len(self.symbols)
                    weights = {sym: 1.0 / n for sym in self.symbols}
                    cash_weight = 0.0
            else:

                n = len(self.symbols)
                weights = {sym: 1.0 / n for sym in self.symbols}
                cash_weight = 0.0


            if self.run_main_council and self._prev_mrc_weights and self._smoothing_alpha < 1.0:
                alpha_base = self._smoothing_alpha
                all_syms = set(list(weights.keys()) + list(self._prev_mrc_weights.keys()))
                smoothed = {}
                for sym in all_syms:
                    new_w  = weights.get(sym, 0.0)
                    prev_w = self._prev_mrc_weights.get(sym, 0.0)
                    if abs(new_w) < abs(prev_w):

                        a = 0.95 if (abs(new_w) < 1e-6 and abs(prev_w) > 1e-6) else min(1.0, alpha_base + 0.08)
                    else:

                        a = max(0.70, alpha_base - 0.05)
                    smoothed[sym] = a * new_w + (1.0 - a) * prev_w
                weights = smoothed

                cash_weight = alpha_base * cash_weight + (1.0 - alpha_base) * self._prev_mrc_cash

                total = sum(weights.values()) + cash_weight
                if abs(total) > 1e-9:
                    weights     = {k: round(v / total, 8) for k, v in weights.items()}
                    cash_weight = round(cash_weight / total, 8)
            if self.run_main_council:
                self._prev_mrc_weights = dict(weights)
                self._prev_mrc_cash    = cash_weight


            #


            #


            #


            #
            #     short_scale = f1 × f2 × f3

            if self.config.allow_short and self.run_main_council:
                _any_short = any(w < 0 for w in weights.values())
                if _any_short:
                    _gate_p7  = self._get_prices_on_date(eval_date - datetime.timedelta(days=7))
                    _gate_p30 = self._get_prices_on_date(eval_date - datetime.timedelta(days=30))
                    stage1_evals = result.get("stage1_evals", {})
                    _scaled = False

                    for sym in list(weights.keys()):
                        if weights[sym] >= 0:
                            continue


                        f1 = max(0.0, -bundle.regime_score)


                        _zero_count = 0
                        for _ak in ["agent1", "agent2", "agent3"]:
                            _ev = stage1_evals.get(_ak)
                            if _ev is not None:
                                _priv = _ev.parsed.get("private_portfolio_weights", {})
                                if float(_priv.get(sym, 1.0)) == 0.0:
                                    _zero_count += 1
                        f2 = _zero_count / 3.0


                        _p1  = (bundle.prices or {}).get(sym)
                        _p7  = (_gate_p7  or {}).get(sym)
                        _p30 = (_gate_p30 or {}).get(sym)
                        if _p1 and _p7 and _p7 > 0 and _p30 and _p30 > 0:
                            r7  = _p1 / _p7  - 1.0
                            r30 = _p1 / _p30 - 1.0
                            f3 = (min(1.0, max(0.0, -r7  / 0.05))
                                  * min(1.0, max(0.0, -r30 / 0.10)))
                        else:
                            f3 = 0.0

                        short_scale = f1 * f2 * f3
                        new_w = weights[sym] * short_scale
                        if abs(new_w) < 1e-6:
                            new_w = 0.0
                        if abs(new_w - weights[sym]) > 1e-8:
                            logger.debug(
                                f"[ShortScale] {eval_date} {sym}: "
                                f"w={weights[sym]:.4f}→{new_w:.4f} "
                                f"(f1={f1:.3f} f2={f2:.3f} f3={f3:.3f})"
                            )
                            weights[sym] = new_w
                            _scaled = True


                    if _scaled:
                        _net = sum(weights.values())
                        _total = _net + cash_weight
                        if abs(_total) > 1e-9:
                            weights     = {k: round(v / _total, 8) for k, v in weights.items()}
                            cash_weight = round(cash_weight / _total, 8)


            prices = bundle.prices
            if not prices:
                logger.warning(f"[Backtest] No prices for {eval_date}, skipping rebalance.")
                continue


            if self._stages_log_path and self.run_main_council and "stage_record" in result:
                _srec = result["stage_record"]

                _sw = {
                    "agent1":  dict(_srec.stage1_weights.get("agent1", {})),
                    "agent2":  dict(_srec.stage1_weights.get("agent2", {})),
                    "agent3":  dict(_srec.stage1_weights.get("agent3", {})),
                    "pair_12": dict(_srec.stage2_weights.get("1_2", {})),
                    "pair_13": dict(_srec.stage2_weights.get("1_3", {})),
                    "pair_23": dict(_srec.stage2_weights.get("2_3", {})),
                    "stage3":  dict(getattr(_srec, "stage3_weights", {}) or {}),
                }
                _sc = {
                    "agent1":  _srec.stage1_cash.get("agent1", 0.0),
                    "agent2":  _srec.stage1_cash.get("agent2", 0.0),
                    "agent3":  _srec.stage1_cash.get("agent3", 0.0),
                    "pair_12": _srec.stage2_cash.get("1_2", 0.0),
                    "pair_13": _srec.stage2_cash.get("1_3", 0.0),
                    "pair_23": _srec.stage2_cash.get("2_3", 0.0),
                    "stage3":  getattr(_srec, "stage3_cash", 0.0) or 0.0,
                }

                _stage_prets: Dict[str, float] = {}
                for _sk in _STAGE_KEYS:
                    if _prev_prices:
                        _pr = 0.0
                        for _sym, _w in _sw[_sk].items():
                            _p0 = _prev_prices.get(_sym)
                            _p1 = prices.get(_sym)
                            if _p0 and _p0 > 0 and _p1 and _p1 > 0:
                                _pr += _w * (_p1 / _p0 - 1.0)
                        _stage_prets[_sk] = round(_pr, 8)
                    else:
                        _stage_prets[_sk] = 0.0
                    _stage_cum[_sk] *= (1.0 + _stage_prets[_sk])

                _stages_records.append({
                    "period":              i + 1,
                    "date":                str(eval_date),
                    "market_regime":       regime_str,
                    "regime_score":        round(regime_score, 6),
                    "stage_weights":       {k: {s: round(v, 6) for s, v in _sw[k].items() if abs(v) > 1e-9}
                                            for k in _STAGE_KEYS},
                    "stage_cash":          {k: round(_sc[k], 6) for k in _STAGE_KEYS},
                    "stage_period_returns": _stage_prets,
                    "stage_cum_values":    {k: round(_stage_cum[k], 8) for k in _STAGE_KEYS},
                })


            if prices:
                _prev_prices = dict(prices)


            #


            #


            if self.run_main_council:
                prev30_prices = self._get_prices_on_date(eval_date - datetime.timedelta(days=30))


                _btc_p30 = (prev30_prices or {}).get("BTC")
                _btc_p0  = (prices or {}).get("BTC")
                if _btc_p30 and _btc_p0 and _btc_p30 > 0:
                    _btc_ret_30d = _btc_p0 / _btc_p30 - 1.0
                    _ew_rets = [
                        prices[s] / prev30_prices[s] - 1
                        for s in self.symbols
                        if s != "BTC"
                        and s in prices
                        and s in prev30_prices
                        and prev30_prices[s] > 0
                    ]
                    _ew_ret_30d = sum(_ew_rets) / len(_ew_rets) if _ew_rets else 0.0
                    _btc_dom = _btc_ret_30d - _ew_ret_30d
                bundle.btc_dominance = _btc_dom

                if prev30_prices and prices:
                    mom_rets = {}
                    for sym in self.symbols:
                        p0 = prev30_prices.get(sym)
                        p1 = prices.get(sym)
                        if p0 and p0 > 0 and p1 and p1 > 0:
                            mom_rets[sym] = p1 / p0 - 1.0
                    if len(mom_rets) >= 3:
                        vals = list(mom_rets.values())
                        mu = sum(vals) / len(vals)
                        std = math.sqrt(
                            sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
                        )
                        if std > 0:

                            z_scores = {
                                sym: math.tanh((r - mu) / std / 1.5)
                                for sym, r in mom_rets.items()
                            }
                            _max_short = self.config.max_short_weight if self.config.allow_short else 0.0


                            _bull_mom = (
                                self._mom_strength if self._mom_strength is not None
                                else 0.08 + 0.35 * max(0.0, bundle.regime_score)
                            )

                            _bear_mom = (
                                0.0 if (self._mom_strength is not None or not self.config.allow_short)
                                else 0.06 * max(0.0, -bundle.regime_score)
                            )


                            weights = {
                                sym: max(
                                    -_max_short if weights.get(sym, 0.0) < 0 else 0.0,
                                    weights.get(sym, 0.0)
                                    * (1.0 + _bull_mom * z_scores.get(sym, 0.0)),
                                )
                                for sym in weights
                            }


                            if _bear_mom > 1e-6:
                                for sym in self.symbols:
                                    z = z_scores.get(sym, 0.0)
                                    if z < 0:
                                        short_adj = _bear_mom * math.tanh(-z / 0.8)
                                        new_w = weights.get(sym, 0.0) - short_adj
                                        weights[sym] = max(-_max_short, new_w)

                            net = sum(weights.values())
                            total = net + cash_weight
                            if abs(net) > 1e-9 and total > 0:
                                weights = {k: round(v / total, 8) for k, v in weights.items()}
                                cash_weight = round(cash_weight / total, 8)

            # 4d. BTC Dominance Overlay


            if self.run_main_council and prev30_prices and prices:
                _dom_signal = math.tanh(_btc_dom / 0.15)

                if _dom_signal > 1e-3 and regime_str in ("volatile", "bear"):


                    _LAGGING = ["ETH", "ADA", "LINK", "DOGE", "XLM", "XRP", "BCH"]
                    _adj = _dom_signal * (0.45 if regime_str == "volatile" else 0.30)  # max reduction per token
                    _shifted = 0.0
                    for _s in _LAGGING:
                        _w = weights.get(_s, 0.0)
                        if _w > 0:
                            _cut = _w * _adj
                            weights[_s] = max(0.0, _w - _cut)
                            _shifted += _cut
                    # 60% → BTC, 25% → TRX, 15% → ZEC
                    weights["BTC"] = min(0.30, weights.get("BTC", 0.0) + _shifted * 0.60)
                    weights["TRX"] = min(0.30, weights.get("TRX", 0.0) + _shifted * 0.25)
                    weights["ZEC"] = min(0.30, weights.get("ZEC", 0.0) + _shifted * 0.15)
                    _d_total = sum(weights.values()) + cash_weight
                    if abs(_d_total) > 1e-9:
                        weights = {k: round(v / _d_total, 8) for k, v in weights.items()}
                        cash_weight = round(cash_weight / _d_total, 8)
                    logger.debug("[BTCDom] %s BTC-season: dom=%.3f signal=%.3f shifted=%.4f",
                                 eval_date, _btc_dom, _dom_signal, _shifted)

                elif _dom_signal < -1e-3 and regime_str == "bull":


                    _ADJ_DOWN = ["BTC", "TRX"]
                    _ADJ_UP   = ["ADA", "XLM", "DOGE", "XRP", "BCH", "ETH"]
                    _adj = (-_dom_signal) * 0.20  # max 20% reduction
                    _shifted = 0.0
                    for _s in _ADJ_DOWN:
                        _w = weights.get(_s, 0.0)
                        if _w > 0:
                            _cut = _w * _adj
                            weights[_s] = max(0.0, _w - _cut)
                            _shifted += _cut
                    _up_total = sum(max(0.0, weights.get(_s, 0.0)) for _s in _ADJ_UP)
                    if _up_total > 1e-9 and _shifted > 0:
                        for _s in _ADJ_UP:
                            _share = max(0.0, weights.get(_s, 0.0)) / _up_total
                            weights[_s] = min(0.30, weights.get(_s, 0.0) + _shifted * _share)
                    _d_total = sum(weights.values()) + cash_weight
                    if abs(_d_total) > 1e-9:
                        weights = {k: round(v / _d_total, 8) for k, v in weights.items()}
                        cash_weight = round(cash_weight / _d_total, 8)
                    logger.debug("[BTCDom] %s alt-season: dom=%.3f signal=%.3f shifted=%.4f",
                                 eval_date, _btc_dom, _dom_signal, _shifted)


            #


            if self.run_main_council and regime_str == "volatile":
                _VOLATILE_BTC_FLOOR = 0.18
                _cur_btc_vf = weights.get("BTC", 0.0)
                if _cur_btc_vf < _VOLATILE_BTC_FLOOR:
                    _lift_vf = _VOLATILE_BTC_FLOOR - _cur_btc_vf
                    _long_alts_vf = {s: w for s, w in weights.items() if s != "BTC" and w > 0}
                    _alts_total_vf = sum(_long_alts_vf.values())
                    if _alts_total_vf > _lift_vf + 0.01:
                        weights["BTC"] = _VOLATILE_BTC_FLOOR
                        for _svf in _long_alts_vf:
                            weights[_svf] = max(
                                0.0,
                                weights[_svf] - _lift_vf * (_long_alts_vf[_svf] / _alts_total_vf),
                            )
                        _vf_total = sum(weights.values()) + cash_weight
                        if abs(_vf_total) > 1e-9:
                            weights = {k: round(v / _vf_total, 8) for k, v in weights.items()}
                            cash_weight = round(cash_weight / _vf_total, 8)
                        logger.info(
                            "[VolatileBTCFloor] %s: BTC %.3f→%.3f (lifted %.3f from alts)",
                            eval_date, _cur_btc_vf, weights.get("BTC", 0.0), _lift_vf,
                        )


            #


            if self.run_main_council and regime_str == "bear":
                _onchain = getattr(bundle, "onchain_metrics", {}) or {}
                if _onchain:
                    _btc_z_vals: List[float] = []
                    _btc_om = _onchain.get("BTC")
                    if _btc_om:
                        _zs = getattr(_btc_om, "extra", {}).get("z_scores", {})
                        _btc_z_vals = [v for k, v in _zs.items()
                                       if k in ("AdrActCnt", "TxCnt") and v is not None]
                    _other_z_avgs: List[float] = []
                    for _sym2, _om2 in _onchain.items():
                        if _sym2 == "BTC":
                            continue
                        _zs2 = getattr(_om2, "extra", {}).get("z_scores", {})
                        _vals2 = [v for k, v in _zs2.items()
                                  if k in ("AdrActCnt", "TxCnt") and v is not None]
                        if _vals2:
                            _other_z_avgs.append(sum(_vals2) / len(_vals2))
                    if _btc_z_vals and _other_z_avgs:
                        _btc_z_avg   = sum(_btc_z_vals) / len(_btc_z_vals)
                        _alt_z_avg   = sum(_other_z_avgs) / len(_other_z_avgs)
                        _onchain_div = _btc_z_avg - _alt_z_avg

                        _btc_tilt = 0.08 * math.tanh(_onchain_div / 1.5)
                        if _btc_tilt > 0.005:
                            _cur_btc = weights.get("BTC", 0.0)
                            _new_btc = min(0.30, _cur_btc + _btc_tilt)
                            _delta_btc = _new_btc - _cur_btc
                            if _delta_btc > 0.001:
                                _long_alts = {s: w for s, w in weights.items()
                                              if s != "BTC" and w > 0}
                                _alts_total = sum(_long_alts.values())
                                if _alts_total > _delta_btc + 0.01:
                                    weights["BTC"] = _new_btc
                                    for _s3 in _long_alts:
                                        weights[_s3] = max(
                                            0.0,
                                            weights[_s3] - _delta_btc
                                            * (_long_alts[_s3] / _alts_total),
                                        )
                                    logger.debug(
                                        "[BearBTCOnchain] %s: btc_z=%.2f alt_z=%.2f "
                                        "div=%.2f tilt=%.3f BTC %.3f→%.3f",
                                        eval_date, _btc_z_avg, _alt_z_avg,
                                        _onchain_div, _btc_tilt, _cur_btc, _new_btc,
                                    )


            #

            #     _target = 0.08 + 0.17 × exp(-|score| / 0.12)


            #

            if self.run_main_council and regime_str in ("volatile", "bull"):
                if regime_str == "volatile":
                    _abs_score = abs(bundle.regime_score)
                    _target_cash = 0.08 + 0.17 * math.exp(-_abs_score / 0.12)
                else:  # bull
                    _target_cash = 0.08

                if cash_weight < _target_cash:

                    _needed = _target_cash - cash_weight
                    _long_total = sum(max(0.0, w) for w in weights.values())
                    if _long_total > 1e-9:
                        for _s in list(weights.keys()):
                            if weights[_s] > 0:
                                weights[_s] = max(
                                    0.0,
                                    weights[_s] - _needed * (weights[_s] / _long_total),
                                )
                    cash_weight = _target_cash
                elif cash_weight > _target_cash:

                    _excess = cash_weight - _target_cash
                    cash_weight = _target_cash
                    _long_total = sum(max(0.0, w) for w in weights.values())
                    if _long_total > 1e-9:
                        for _s in weights:
                            if weights[_s] > 0:
                                weights[_s] += _excess * (weights[_s] / _long_total)

                _ce_total = sum(weights.values()) + cash_weight
                if abs(_ce_total) > 1e-9:
                    weights = {k: round(v / _ce_total, 8) for k, v in weights.items()}
                    cash_weight = round(cash_weight / _ce_total, 8)
                logger.info(
                    "[CashTarget] %s %s: |score|=%.3f target=%.3f actual=%.3f",
                    eval_date, regime_str, abs(bundle.regime_score), _target_cash, cash_weight,
                )


            if self.run_main_council and _bull_to_volatile:
                _score_drop = max(0.0, self._prev_smoothed_score - bundle.regime_score)
                _bt_scale = 1.0 - 0.35 * math.tanh(_score_drop / 0.25)
                _bt_freed = 0.0
                _bt_weights: Dict[str, float] = {}
                for _sym, _w in weights.items():
                    if _w > 0:
                        _bt_freed += _w * (1.0 - _bt_scale)
                        _bt_weights[_sym] = _w * _bt_scale
                    else:
                        _bt_weights[_sym] = _w
                weights = _bt_weights
                cash_weight = min(0.30, cash_weight + _bt_freed)
                _bt_total = sum(weights.values()) + cash_weight
                if abs(_bt_total) > 1e-9:
                    weights = {k: round(v / _bt_total, 8) for k, v in weights.items()}
                    cash_weight = round(cash_weight / _bt_total, 8)
                logger.info(
                    "[TransitionBuffer] %s bull→volatile: scaled ×0.80, freed=%.4f to cash",
                    eval_date, _bt_freed,
                )


            #

            #       regime_gate = max(0, -regime_score)


            #
            #     dd_scale = 1.0 - regime_gate × tanh(current_dd / 0.15) × 0.40


            #


            if self.run_main_council and prev_value is not None and prev_value > 0:
                if prev_value > self._peak_portfolio_value:
                    self._peak_portfolio_value = prev_value
                if self._peak_portfolio_value > 0:
                    current_dd = max(0.0, (self._peak_portfolio_value - prev_value) / self._peak_portfolio_value)
                    regime_gate = max(0.0, -bundle.regime_score)
                    if regime_gate > 1e-6 and current_dd > 1e-6:
                        dd_scale = 1.0 - regime_gate * math.tanh(current_dd / 0.15) * 0.40
                        cash_boost = 0.0
                        scaled_weights = {}
                        for sym, w in weights.items():
                            if w > 0:
                                scaled_weights[sym] = w * dd_scale
                                cash_boost += w * (1.0 - dd_scale)
                            else:
                                scaled_weights[sym] = w
                        weights = scaled_weights
                        max_cash = 0.08 + 0.22 * regime_gate
                        cash_weight = min(max_cash, cash_weight + cash_boost)
                        net = sum(weights.values())
                        total = net + cash_weight
                        if abs(net) > 1e-9 and total > 0:
                            weights = {k: round(v / total, 8) for k, v in weights.items()}
                            cash_weight = round(cash_weight / total, 8)
                        logger.debug(
                            f"[DD-Protect] {eval_date}: dd={current_dd:.1%} "
                            f"gate={regime_gate:.2f} scale={dd_scale:.3f} "
                            f"max_cash={max_cash:.2f} cash={cash_weight:.3f}"
                        )


            record = self.portfolio.rebalance(weights, cash_weight, prices, eval_date)
            turnovers.append(record.turnover)


            curr_value = record.post_value

            if self.run_main_council and curr_value > self._peak_portfolio_value:
                self._peak_portfolio_value = curr_value
            if prev_value is not None and prev_value > 0:
                ret = (curr_value / prev_value) - 1.0
                period_returns.append(ret)


                if self.run_main_council:
                    self.council.record_period_result(
                        date=eval_date,
                        actual_return=ret,
                        portfolio_value=curr_value,
                        prediction=result,
                    )
            prev_value = curr_value


            _bl_period: Dict[str, Any] = {}
            for bl_name, bl_council in self.baseline_councils.items():
                try:
                    bl_result = await bl_council.run_with_bundle(bundle)
                    bl_weights   = bl_result.get("portfolio_weights", {})
                    bl_cash      = bl_result.get("cash_weight", 0.0)
                except Exception as e:
                    logger.warning(
                        "[Backtest] Baseline '%s' failed on %s (%s: %s). Using equal weight.",
                        bl_name, eval_date, type(e).__name__, e,
                    )
                    n = len(self.symbols)
                    bl_weights = {sym: 1.0 / n for sym in self.symbols}
                    bl_cash = 0.0

                bl_record = self.baseline_portfolios[bl_name].rebalance(
                    bl_weights, bl_cash, prices, eval_date
                )
                self._baseline_turnovers[bl_name].append(bl_record.turnover)

                bl_prev = self._baseline_prev_vals[bl_name]
                bl_curr = bl_record.post_value
                bl_ret_this: Optional[float] = None
                if bl_prev is not None and bl_prev > 0:
                    bl_ret_this = (bl_curr / bl_prev) - 1.0
                    self._baseline_returns[bl_name].append(bl_ret_this)
                    if hasattr(bl_council, "record_period_result"):
                        bl_council.record_period_result(
                            date=eval_date,
                            actual_return=bl_ret_this,
                            portfolio_value=bl_curr,
                        )
                self._baseline_prev_vals[bl_name] = bl_curr
                _bl_period[bl_name] = {
                    "total_value":   round(bl_curr, 2),
                    "period_return": round(bl_ret_this, 8) if bl_ret_this is not None else None,
                }


            if _inc_file is not None:
                _period_ret = period_returns[-1] if period_returns else None
                _period_rec = {
                    "period":        i + 1,
                    "date":          str(eval_date),
                    "total_value":   round(curr_value, 2),
                    "period_return": round(_period_ret, 8) if _period_ret is not None else None,
                    "weights":       {k: round(v, 8) for k, v in weights.items() if abs(v) > 1e-9},
                    "cash_weight":   round(cash_weight, 8),
                    "market_regime": bundle.market_regime,
                    "regime_score":  round(bundle.regime_score, 6),
                    "turnover":      round(record.turnover, 8),
                }
                if "shapley_snapshot" in result:
                    _period_rec["shapley_snapshot"] = result["shapley_snapshot"]
                if _bl_period:
                    _period_rec["baselines"] = _bl_period
                if not _inc_first:
                    _inc_file.write(",\n")
                else:
                    _inc_first = False
                _inc_file.write(json.dumps(_period_rec, default=str))
                _inc_file.flush()


        if _inc_file is not None:
            _inc_file.write("\n]")
            _inc_file.close()
            logger.info("[Backtest] Incremental JSON saved → %s", self._incremental_json_path)


        if self._stages_log_path and _stages_records:
            _stages_dir = os.path.dirname(self._stages_log_path)
            if _stages_dir:
                os.makedirs(_stages_dir, exist_ok=True)

            _final_cum = {k: round((_stage_cum[k] - 1.0) * 100, 4) for k in _STAGE_KEYS}
            _stages_output = {
                "summary": {
                    "total_periods": len(_stages_records),
                    "start_date": _stages_records[0]["date"] if _stages_records else None,
                    "end_date":   _stages_records[-1]["date"] if _stages_records else None,
                    "final_cumulative_return_pct": _final_cum,
                },
                "records": _stages_records,
            }
            with open(self._stages_log_path, "w", encoding="utf-8") as _sf:
                json.dump(_stages_output, _sf, indent=2, default=str)
            logger.info("[Backtest] Stages log saved → %s", self._stages_log_path)

            logger.info(
                "[StagesSummary] Final cumulative returns (%%):\n"
                "  Agent1=%.1f%%  Agent2=%.1f%%  Agent3=%.1f%%\n"
                "  Pair12=%.1f%%  Pair13=%.1f%%  Pair23=%.1f%%\n"
                "  Stage3(MRC)=%.1f%%",
                _final_cum["agent1"], _final_cum["agent2"], _final_cum["agent3"],
                _final_cum["pair_12"], _final_cum["pair_13"], _final_cum["pair_23"],
                _final_cum["stage3"],
            )


        shapley_report = ""
        if hasattr(self.council, "shapley_ledger"):

            if bundle is not None:
                self.council.shapley_ledger.finalize(bundle.prices)
            shapley_report = self.council.shapley_ledger.format_shapley_report(
                for_agent4_prompt=False
            )
            logger.info("[Backtest] ShapleyCouncil final Shapley report:\n%s", shapley_report)


        logger.info("[Backtest] Computing metrics...")
        mrc_label = "MRC-Agent" if self.run_main_council else "EqualW(placeholder)"
        mrc_metrics = compute_all_metrics(
            portfolio_returns=period_returns,
            turnovers=turnovers,
            risk_free=self.config.risk_free_rate,
            freq=365,
            label=mrc_label,
        )


        baselines = self._compute_baselines(eval_dates)


        for bl_name in self.baseline_councils:
            bl_rets = self._baseline_returns[bl_name]
            bl_tos  = self._baseline_turnovers[bl_name]
            if bl_rets:
                bl_metrics = compute_all_metrics(
                    portfolio_returns=bl_rets,
                    turnovers=bl_tos,
                    risk_free=self.config.risk_free_rate,
                    freq=365,
                    label=bl_name,
                )
                bl_metrics["period_returns"] = bl_rets
                baselines[bl_name] = bl_metrics
                logger.info(
                    "[Backtest] %s: Sharpe=%.2f Cumret=%.1f%%",
                    bl_name,
                    bl_metrics.get("sharpe_ratio", 0),
                    (bl_metrics.get("cumulative_return", 0) or 0) * 100,
                )


        ew_returns = baselines.get("equal_weight", {}).get("period_returns", [])
        if ew_returns:
            from .metrics import information_ratio
            mrc_metrics["information_ratio"] = round(
                information_ratio(period_returns, ew_returns, freq=365), 4
            )

        config_summary = {
            "start_date": str(self.config.start_date),
            "end_date": str(self.config.end_date),
            "rebalance_freq_days": self.config.rebalance_freq_days,
            "transaction_cost_bps": self.config.transaction_cost_bps,
            "lookback_days": self.lookback_days,
            "kline_style": self.kline_style,
            "n_tokens": len(self.symbols),
            "max_weight": self.config.max_weight,
            "max_cash": self.config.max_cash,
            "initial_capital": 100_000.0,
            "run_main_council": self.run_main_council,
        }

        result = BacktestResult(
            portfolio_history=self.portfolio.portfolio_history,
            period_returns=period_returns,
            turnovers=turnovers,
            metrics=mrc_metrics,
            baselines=baselines,
            config_summary=config_summary,
            shapley_history=shapley_history,
            shapley_report=shapley_report,
        )

        logger.info("[Backtest] Complete.")
        result.print_report()
        return result

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def _fetch_date_range(
        self,
        start_date: datetime.date,
        eval_date: datetime.date,
    ) -> DateRangeBundle:
        """"""
        dm = self.data_manager

        # ---- Agent 1 ----
        a1 = dm.get_agent1_data(self.symbols, start_date, eval_date)

        # ---- Agent 2 ----
        a2 = dm.get_agent2_data(self.symbols, start_date, eval_date)

        # ---- Agent 3 ----
        a3 = dm.get_agent3_data(start_date, eval_date)

        return DateRangeBundle(
            eval_date=eval_date,
            market_snapshots=a1["market_snapshots"],
            kline_images=a1["kline_images"],
            onchain_metrics=a2["onchain_metrics"],
            btc_macro_df=a2["btc_macro_df"],
            env_snapshot=a3["env_snapshot"],
            fred_df=a3["fred_df"],
            fng_df=a3["fng_df"],
            polymarket_df=a3["polymarket_df"],
            tweets_df=a3["tweets_df"],
            prices=a1["prices"],
        )

    # ------------------------------------------------------------------
    #  Baselines
    # ------------------------------------------------------------------

    def _compute_baselines(self, eval_dates: List[datetime.date]) -> Dict[str, Any]:
        """"""
        baselines = {}

        # ---- Equal Weight (1/N) ----
        ew_returns = self._compute_equal_weight_returns(eval_dates)
        if ew_returns:
            ew_turnovers = [0.0] * len(ew_returns)
            ew_metrics = compute_all_metrics(
                portfolio_returns=ew_returns,
                turnovers=ew_turnovers,
                freq=365,
                label="equal_weight",
            )
            ew_metrics["period_returns"] = ew_returns
            baselines["equal_weight"] = ew_metrics

        # ---- BTC Hold ----
        btc_returns = self._compute_btc_hold_returns(eval_dates)
        if btc_returns:
            btc_metrics = compute_all_metrics(
                portfolio_returns=btc_returns,
                turnovers=[0.0] * len(btc_returns),
                freq=365,
                label="btc_hold",
            )
            btc_metrics["period_returns"] = btc_returns
            baselines["btc_hold"] = btc_metrics

        return baselines

    def _compute_equal_weight_returns(self, eval_dates: List[datetime.date]) -> List[float]:
        """"""
        returns = []
        prev_prices: Optional[Dict[str, float]] = None

        for date in eval_dates:
            prices = self._get_prices_on_date(date)
            if prev_prices and prices:
                common = [s for s in self.symbols if s in prices and s in prev_prices and prev_prices[s] > 0]
                if common:
                    avg_ret = sum((prices[s] / prev_prices[s] - 1) for s in common) / len(common)
                    returns.append(avg_ret)
            prev_prices = prices

        return returns

    def _compute_btc_hold_returns(self, eval_dates: List[datetime.date]) -> List[float]:
        """"""
        returns = []
        prev_price: Optional[float] = None

        for date in eval_dates:
            prices = self._get_prices_on_date(date)
            btc_price = prices.get("BTC")
            if btc_price and prev_price and prev_price > 0:
                returns.append(btc_price / prev_price - 1.0)
            prev_price = btc_price

        return returns

    def _get_prices_on_date(self, date: datetime.date) -> Dict[str, float]:
        """"""

        for rec in reversed(self.portfolio.rebalance_history):
            if rec.date == date and rec.prices:
                return rec.prices


        prices: Dict[str, float] = {}
        reader = self.data_manager.historical
        for sym in self.symbols:
            p = reader.get_price(sym, date)
            if p is not None:
                prices[sym] = p
        return prices

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _detect_market_regime(
        self,
        eval_date: datetime.date,
        current_prices: Dict[str, float],
    ) -> Tuple[str, float]:
        """"""
        reader = self.data_manager.historical


        _BASKET = {"BTC": 0.50, "ETH": 0.30, "BNB": 0.20}


        basket_now_prices: Dict[str, float] = {
            sym: current_prices.get(sym, 0.0) for sym in _BASKET
        }

        def _basket_ret(days_ago: int) -> Optional[float]:
            """"""
            wret = 0.0
            valid_w = 0.0
            for sym, wt in _BASKET.items():
                p_now = basket_now_prices.get(sym, 0.0)
                p_old = reader.get_price(sym, eval_date - datetime.timedelta(days=days_ago)) or 0.0
                if p_now > 0 and p_old > 0:
                    wret   += wt * (p_now / p_old - 1.0)
                    valid_w += wt
            return wret / valid_w if valid_w > 0.5 else None


        basket_weekly: List[float] = []
        for i in range(4):
            ret = _basket_ret((i + 1) * 7)
            prev_ret = _basket_ret(i * 7) if i > 0 else 0.0

            p_this_week: Dict[str, float] = {}
            p_last_week: Dict[str, float] = {}
            wret_week = 0.0; valid_w = 0.0
            for sym, wt in _BASKET.items():
                p_now = reader.get_price(sym, eval_date - datetime.timedelta(days=i * 7)) or 0.0
                p_ago = reader.get_price(sym, eval_date - datetime.timedelta(days=(i + 1) * 7)) or 0.0
                if p_now > 0 and p_ago > 0:
                    wret_week += wt * (p_now / p_ago - 1.0)
                    valid_w   += wt
            if valid_w > 0.5:
                basket_weekly.append(wret_week / valid_w)

        vol_monthly: Optional[float] = None
        if len(basket_weekly) >= 3:
            mu    = sum(basket_weekly) / len(basket_weekly)
            std_w = math.sqrt(
                sum((r - mu) ** 2 for r in basket_weekly) / max(len(basket_weekly) - 1, 1)
            )
            vol_monthly = max(std_w * math.sqrt(4.0), 0.03)


        basket_30d_ret = _basket_ret(30)
        basket_7d_ret  = _basket_ret(7)


        syn_now = 1.0
        syn_30d = (1.0 / (1.0 + basket_30d_ret)) if basket_30d_ret is not None and basket_30d_ret > -1.0 else None
        syn_7d  = (1.0 / (1.0 + basket_7d_ret))  if basket_7d_ret  is not None and basket_7d_ret  > -1.0 else None

        regime, score = self._regime_detector.detect(
            btc_price_now     = syn_now,
            btc_price_30d_ago = syn_30d,
            btc_price_7d_ago  = syn_7d,
            vol_monthly       = vol_monthly,
        )
        return regime, score

    def _compute_transition_boost(
        self,
        new_regime: str,
    ) -> Optional[Dict[str, float]]:
        """"""
        if self._prev_regime is None:

            self._prev_regime = new_regime
            return None

        boost: Optional[Dict[str, float]] = None

        if new_regime != self._prev_regime:

            if new_regime == "bull":
                self._transition_boost_agent = "agent1"
                boost = {"agent1": 1.25}
            elif new_regime == "bear":
                self._transition_boost_agent = "agent3"
                boost = {"agent3": 1.25}
            else:  # volatile
                if self._prev_regime == "bull":
                    self._transition_boost_agent = "agent1"
                    boost = {"agent1": 1.15}
                else:
                    self._transition_boost_agent = "agent2"
                    boost = {"agent2": 1.15}

            self._transition_remaining = max(1, round(7 / max(1, self._freq_days)))
        else:

            if self._transition_remaining > 0:
                self._transition_remaining -= 1


        self._prev_regime = new_regime
        return boost

    @staticmethod
    def _rebalance_dates(
        start: datetime.date,
        end: datetime.date,
        freq_days: int = 1,
    ) -> List[datetime.date]:
        """"""
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += datetime.timedelta(days=freq_days)
        return dates
