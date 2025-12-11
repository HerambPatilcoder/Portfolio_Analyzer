from typing import Dict, Any

import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns


def run_optimizations(
    price_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Run standard PyPortfolioOpt optimizations on the given price data.

    Returns a dict with entries for:
        - "max_sharpe"
        - "min_vol"

    Each entry contains:
        - weights: dict[ticker -> weight]
        - ret: float (annualized)
        - vol: float (annualized)
        - sharpe: float
    """

    # Expected returns & covariance matrix
    mu = expected_returns.mean_historical_return(price_df)      # annualized
    S = risk_models.sample_cov(price_df)                        # annualized cov

    results = {}

    # ----- Max Sharpe Ratio Portfolio -----
    ef_ms = EfficientFrontier(mu, S)
    w_ms = ef_ms.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_w_ms = ef_ms.clean_weights()
    ret_ms, vol_ms, sharpe_ms = ef_ms.portfolio_performance(risk_free_rate=risk_free_rate)

    results["max_sharpe"] = {
        "weights": cleaned_w_ms,
        "ret": ret_ms,
        "vol": vol_ms,
        "sharpe": sharpe_ms,
    }

    # ----- Minimum Volatility Portfolio -----
    ef_mv = EfficientFrontier(mu, S)
    w_mv = ef_mv.min_volatility()
    cleaned_w_mv = ef_mv.clean_weights()
    ret_mv, vol_mv, sharpe_mv = ef_mv.portfolio_performance(risk_free_rate=risk_free_rate)

    results["min_vol"] = {
        "weights": cleaned_w_mv,
        "ret": ret_mv,
        "vol": vol_mv,
        "sharpe": sharpe_mv,
    }

    return results