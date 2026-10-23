""""""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BacktestResult:
    """"""
    portfolio_history: List[Dict[str, Any]] = field(default_factory=list)
    period_returns: List[float] = field(default_factory=list)
    turnovers: List[float] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    shapley_history: List[Dict[str, Any]] = field(default_factory=list)
    shapley_report: str = ""

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "metrics": self.metrics,
            "baselines": self.baselines,
            "config_summary": self.config_summary,
            "period_returns": self.period_returns,
            "turnovers": self.turnovers,
            "shapley_report": self.shapley_report,
            "shapley_history": self.shapley_history,
            "portfolio_history": [
                {k: str(v) if isinstance(v, datetime.date) else v for k, v in rec.items()}
                for rec in self.portfolio_history
            ],
        }

    def print_report(self):
        """"""
        start = self.config_summary.get("start_date", "N/A")
        end = self.config_summary.get("end_date", "N/A")
        freq = self.config_summary.get("rebalance_freq_days", 7)
        tc = self.config_summary.get("transaction_cost_bps", 10)

        print("\n" + "=" * 60)
        print("         Backtest Report — Market Regime Council")
        print("=" * 60)
        print(f"Period   : {start} → {end}")
        print(f"Rebalance: Every {freq} days | TC: {tc} bps (one-way)")
        print(f"Tokens   : {self.config_summary.get('n_tokens', 13)} tokens")
        print("-" * 60)


        metrics_rows = [
            ("Cumul. Return",    "cumulative_return",    lambda v: f"{v:+.1%}"),
            ("Ann. Return",      "annualized_return",    lambda v: f"{v:+.1%}"),
            ("Sharpe Ratio",     "sharpe_ratio",         lambda v: f"{v:.2f}"),
            ("Sortino Ratio",    "sortino_ratio",        lambda v: f"{v:.2f}"),
            ("Max Drawdown",     "max_drawdown",         lambda v: f"-{v:.1%}"),
            ("Calmar Ratio",     "calmar_ratio",         lambda v: f"{v:.2f}"),
            ("Ann. Turnover",    "annualized_turnover",  lambda v: f"{v:.2f}x"),
            ("Win Rate",         "win_rate",             lambda v: f"{v:.1%}"),
            ("Ann. Volatility",  "annualized_volatility",lambda v: f"{v:.1%}"),
            ("Info Ratio (1/N)", "information_ratio",    lambda v: f"{v:.2f}" if v is not None else "N/A"),
        ]


        ew  = self.baselines.get("equal_weight", {})
        btc = self.baselines.get("btc_hold", {})


        extra_baseline_keys = [
            k for k in self.baselines
            if k not in ("equal_weight", "btc_hold")
        ]

        _alias = {
            "fincon":          "FinCon",
            "trading_agents":  "TradingAgt",
            "finmem":          "FinMem",
            "cryptotrade":     "CryptoTrd",
            "fs_reasoning":    "FS-Reason",
            "a2c":             "A2C",
            "ppo":             "PPO",
            "dqn":             "DQN",
            "mappo":           "MAPPO",
            "maddpg":          "MADDPG",
        }
        extra_labels = [_alias.get(k, k[:10]) for k in extra_baseline_keys]


        fixed_cols  = f"{'MRC-Agent':>12} | {'Equal-W':>10} | {'BTC-Hold':>10}"
        extra_cols  = "".join(f" | {lbl:>11}" for lbl in extra_labels)
        header = f"{'Metric':<20} | {fixed_cols}{extra_cols}"
        print(header)
        print("-" * len(header))

        for label, key, fmt in metrics_rows:
            mrc_v = self.metrics.get(key)
            ew_v  = ew.get(key)
            btc_v = btc.get(key)
            try:
                mrc_str = fmt(mrc_v) if mrc_v is not None else "N/A"
            except Exception:
                mrc_str = "N/A"
            try:
                ew_str = fmt(ew_v) if ew_v is not None else "--"
            except Exception:
                ew_str = "--"
            try:
                btc_str = fmt(btc_v) if btc_v is not None else "--"
            except Exception:
                btc_str = "--"
            row = f"{label:<20} | {mrc_str:>12} | {ew_str:>10} | {btc_str:>10}"
            for bl_key in extra_baseline_keys:
                bl_v = self.baselines[bl_key].get(key)
                try:
                    bl_str = fmt(bl_v) if bl_v is not None else "--"
                except Exception:
                    bl_str = "--"
                row += f" | {bl_str:>11}"
            print(row)

        print("=" * len(header))
        print(f"Periods        : {self.metrics.get('n_periods', len(self.period_returns))}")
        print()


        if self.shapley_report:
            print(self.shapley_report)
            print()

    def plot_cumulative_returns(self, save_path: str = None):
        """"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mticker
        except ImportError:
            print("[BacktestResult] matplotlib not installed. Skipping plot.")
            return


        def _cum_series(returns: List[float]) -> List[float]:
            vals = [1.0]
            for r in returns:
                vals.append(vals[-1] * (1.0 + r))
            return vals

        def _max_dd_series(cum: List[float]) -> List[float]:
            """"""
            peak = cum[0]
            dd = []
            for v in cum:
                if v > peak:
                    peak = v
                dd.append((peak - v) / peak * 100)
            return dd


        has_mrc = self.config_summary.get("run_main_council", True) and bool(self.period_returns)


        raw_dates = [rec.get("date") for rec in self.portfolio_history]
        parsed_dates: List[Optional[datetime.date]] = []
        for d in raw_dates:
            try:
                parsed_dates.append(datetime.date.fromisoformat(str(d)))
            except Exception:
                parsed_dates.append(None)


        if has_mrc:
            mrc_cum = _cum_series(self.period_returns)
            n_pts   = len(mrc_cum)
        else:

            mrc_cum = None
            first_bl_wr = next(
                (v.get("period_returns") for v in self.baselines.values()
                 if v.get("period_returns")),
                [],
            )
            n_pts = len(first_bl_wr) + 1  # +1 to match _cum_series length

        dates_plot = parsed_dates[:n_pts]
        use_dates  = all(d is not None for d in dates_plot) and len(dates_plot) == n_pts


        _styles = {
            # MRC
            "mrc":            dict(color="#2563EB", lw=2.5,  ls="-",            zorder=5, label="MRC-Agent (Ours)"),
            # Price-based baselines
            "equal_weight":   dict(color="#F59E0B", lw=1.5,  ls="--",           zorder=3, label="Equal-Weight (1/N)"),
            "btc_hold":       dict(color="#EAB308", lw=1.5,  ls=":",            zorder=3, label="BTC Buy-and-Hold"),
            # LLM baselines
            "fincon":         dict(color="#16A34A", lw=1.5,  ls="-.",           zorder=4, label="FinCon (NeurIPS 2024)"),
            "trading_agents": dict(color="#DC2626", lw=1.5,  ls="-.",           zorder=4, label="TradingAgents (ICML 2025 WS)"),
            "finmem":         dict(color="#9333EA", lw=1.5,  ls="-.",           zorder=4, label="FinMem (AAAI 2024)"),
            "cryptotrade":    dict(color="#EA580C", lw=1.5,  ls=(0, (6, 2)),    zorder=4, label="CryptoTrade (EMNLP 2024)"),
            "fs_reasoning":   dict(color="#0F766E", lw=1.5,  ls=(0, (6, 2, 1, 2)), zorder=4, label="FS-ReasoningAgent (ICLR 2025 WS)"),
            # DRL baselines
            "a2c":            dict(color="#0891B2", lw=1.2,  ls=(0, (4, 2)),         zorder=3, label="A2C (DRL)"),
            "ppo":            dict(color="#0D9488", lw=1.2,  ls=(0, (4, 2, 1, 2)),   zorder=3, label="PPO (DRL)"),
            "dqn":            dict(color="#D97706", lw=1.2,  ls=(0, (3, 1)),         zorder=3, label="DQN (DRL)"),
            # Multi-agent DRL baselines
            "mappo":          dict(color="#7C3AED", lw=1.2,  ls=(0, (5, 1, 1, 1)),   zorder=3, label="MAPPO (MA-DRL)"),
            "maddpg":         dict(color="#BE185D", lw=1.2,  ls=(0, (2, 1, 2, 1)),   zorder=3, label="MADDPG (MA-DRL)"),
        }


        _bl_order = [
            "equal_weight", "btc_hold",
            "fincon", "trading_agents", "finmem", "cryptotrade", "fs_reasoning",
            "a2c", "ppo", "dqn", "mappo", "maddpg",
        ]


        if has_mrc:
            fig, (ax_ret, ax_dd) = plt.subplots(
                2, 1, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
                sharex=True,
            )
        else:
            fig, ax_ret = plt.subplots(1, 1, figsize=(14, 6))
            ax_dd = None

        def _x(n):
            return dates_plot[:n] if use_dates else list(range(n))


        if has_mrc:
            ax_ret.plot(
                _x(n_pts),
                [(v - 1) * 100 for v in mrc_cum],
                **_styles["mrc"],
            )

        # baselines (price-based + LLM + DRL)
        for bl_name in _bl_order:
            if bl_name not in self.baselines:
                continue
            bl_wr = self.baselines[bl_name].get("period_returns")
            if not bl_wr:
                continue
            bl_cum = _cum_series(bl_wr)
            n_bl   = min(len(bl_cum), n_pts)
            sty    = _styles.get(bl_name, dict(color="gray", lw=1.2, ls="-.", zorder=2,
                                               label=bl_name))
            ax_ret.plot(
                _x(n_bl),
                [(v - 1) * 100 for v in bl_cum[:n_bl]],
                **sty,
            )


        ax_ret.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax_ret.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax_ret.set_title("Market Regime Council — Portfolio Cumulative Returns", fontsize=13, fontweight="bold")
        ax_ret.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax_ret.grid(True, alpha=0.25, linestyle=":")
        ax_ret.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

        if has_mrc:

            final_ret = (mrc_cum[-1] - 1) * 100
            ax_ret.annotate(
                f"MRC: {final_ret:+.1f}%",
                xy=(_x(n_pts)[-1], final_ret),
                xytext=(-60, -15),
                textcoords="offset points",
                color=_styles["mrc"]["color"],
                fontsize=8.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_styles["mrc"]["color"], lw=1),
            )


            mrc_m = self.metrics
            ann_text = (
                f"MRC Performance\n"
                f"──────────────\n"
                f"CumRet:  {mrc_m.get('cumulative_return',0):+.1%}\n"
                f"Sharpe:  {mrc_m.get('sharpe_ratio',0):.2f}\n"
                f"MDD:    -{mrc_m.get('max_drawdown',0):.1%}\n"
                f"WinRate: {mrc_m.get('win_rate',0):.1%}"
            )
            ax_ret.text(
                0.995, 0.97, ann_text,
                transform=ax_ret.transAxes,
                verticalalignment="top", horizontalalignment="right",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )


            mrc_dd = _max_dd_series(mrc_cum)
            ax_dd.fill_between(
                _x(n_pts), 0, [-d for d in mrc_dd],
                color="#2563EB", alpha=0.25, label="MRC Drawdown",
            )
            ax_dd.plot(
                _x(n_pts), [-d for d in mrc_dd],
                color="#2563EB", lw=1.2,
            )
            ax_dd.axhline(0, color="black", lw=0.6, alpha=0.4)
            ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
            ax_dd.grid(True, alpha=0.25, linestyle=":")
            ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))


        x_axis = ax_dd if (has_mrc and ax_dd is not None) else ax_ret
        if use_dates:
            x_axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            x_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(x_axis.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        else:
            x_axis.set_xlabel("Days", fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[BacktestResult] Plot saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)
