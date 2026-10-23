""""""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def cumulative_return(returns: List[float]) -> float:
    """"""
    if not returns:
        return 0.0
    value = 1.0
    for r in returns:
        value *= (1.0 + r)
    return value - 1.0


def annualized_return(returns: List[float], freq: int = 52) -> float:
    """"""
    if not returns:
        return 0.0
    cr = cumulative_return(returns)
    T = len(returns)
    if T == 0:
        return 0.0
    return (1.0 + cr) ** (freq / T) - 1.0


def sharpe_ratio(returns: List[float], risk_free: float = 0.0, freq: int = 52) -> float:
    """"""
    if len(returns) < 2:
        return 0.0
    rf_period = (1.0 + risk_free) ** (1.0 / freq) - 1.0
    excess = [r - rf_period for r in returns]
    mu = sum(excess) / len(excess)
    var = sum((e - mu) ** 2 for e in excess) / (len(excess) - 1)
    sigma = math.sqrt(var) if var > 0 else 0.0
    return (mu / sigma * math.sqrt(freq)) if sigma > 0 else 0.0


def sortino_ratio(returns: List[float], risk_free: float = 0.0, freq: int = 52) -> float:
    """"""
    if len(returns) < 2:
        return 0.0
    rf_period = (1.0 + risk_free) ** (1.0 / freq) - 1.0
    excess = [r - rf_period for r in returns]
    mu = sum(excess) / len(excess)
    T = len(excess)

    var_down = sum(e ** 2 for e in excess if e < 0) / T
    sigma_down = math.sqrt(var_down) if var_down > 0 else 0.0
    if sigma_down == 0.0:
        return float("inf") if mu > 0 else 0.0
    return mu / sigma_down * math.sqrt(freq)


def max_drawdown(returns: List[float]) -> float:
    """"""
    if not returns:
        return 0.0
    value = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        value *= (1.0 + r)
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return mdd


def calmar_ratio(returns: List[float], freq: int = 52) -> float:
    """
    Calmar Ratio: Ann.Return / MDD
    """
    mdd = max_drawdown(returns)
    if mdd <= 0:
        return float("inf")
    ann_ret = annualized_return(returns, freq)
    return ann_ret / mdd


def annualized_turnover(turnovers: List[float], freq: int = 52) -> float:
    """"""
    if not turnovers:
        return 0.0
    return sum(turnovers) / len(turnovers) * freq


def win_rate(returns: List[float]) -> float:
    """"""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)


def information_ratio(returns: List[float], benchmark_returns: List[float],
                      freq: int = 52) -> float:
    """
    Information Ratio vs benchmark: E[r - r_bench] / σ[r - r_bench] × √freq
    """
    n = min(len(returns), len(benchmark_returns))
    if n < 2:
        return 0.0
    active = [returns[i] - benchmark_returns[i] for i in range(n)]
    mu = sum(active) / n
    var = sum((a - mu) ** 2 for a in active) / (n - 1)
    sigma = math.sqrt(var) if var > 0 else 0.0
    return (mu / sigma * math.sqrt(freq)) if sigma > 0 else 0.0


def volatility(returns: List[float], freq: int = 52) -> float:
    """"""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var * freq) if var > 0 else 0.0


def compute_all_metrics(
    portfolio_returns: List[float],
    turnovers: List[float],
    benchmark_returns: Optional[List[float]] = None,
    risk_free: float = 0.0,
    freq: int = 52,
    label: str = "portfolio",
) -> Dict[str, Any]:
    """"""
    cr = cumulative_return(portfolio_returns)
    ann_ret = annualized_return(portfolio_returns, freq)
    sr = sharpe_ratio(portfolio_returns, risk_free, freq)
    so = sortino_ratio(portfolio_returns, risk_free, freq)
    mdd = max_drawdown(portfolio_returns)
    calmar = calmar_ratio(portfolio_returns, freq)
    ann_to = annualized_turnover(turnovers, freq)
    wr = win_rate(portfolio_returns)
    vol = volatility(portfolio_returns, freq)

    ir = None
    if benchmark_returns:
        ir = information_ratio(portfolio_returns, benchmark_returns, freq)

    result = {
        "label": label,
        "n_periods": len(portfolio_returns),
        "cumulative_return": round(cr, 6),
        "annualized_return": round(ann_ret, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(so, 4),
        "max_drawdown": round(mdd, 6),
        "calmar_ratio": round(calmar, 4),
        "annualized_turnover": round(ann_to, 4),
        "win_rate": round(wr, 4),
        "annualized_volatility": round(vol, 6),
    }
    if ir is not None:
        result["information_ratio"] = round(ir, 4)
    return result
