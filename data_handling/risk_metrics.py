import numpy as np
import pandas as pd


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    """
    daily_returns: Series of portfolio daily returns
    risk_free_rate: annual risk-free rate (e.g. 0.02 for 2%)
    """
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()

    if std_daily == 0 or np.isnan(std_daily):
        return np.nan

    excess_return_daily = mean_daily - (risk_free_rate / periods_per_year)
    sharpe = (excess_return_daily / std_daily) * np.sqrt(periods_per_year)
    return sharpe


def compute_drawdown(daily_returns: pd.Series):
    """
    Returns:
        drawdown: Series of drawdown values over time
        max_drawdown: minimum drawdown (most negative)
    """
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95):
    """
    Historical VaR: loss level not exceeded with given confidence.
    Returns a negative number (e.g. -0.03 = -3%).
    """
    if len(daily_returns) == 0:
        return np.nan
    return np.percentile(daily_returns, (1 - confidence) * 100)


def conditional_var(daily_returns: pd.Series, confidence: float = 0.95):
    """
    Conditional VaR (Expected Shortfall): average loss given that
    the loss is worse than VaR.
    """
    var = value_at_risk(daily_returns, confidence)
    if np.isnan(var):
        return np.nan
    tail_losses = daily_returns[daily_returns <= var]
    if len(tail_losses) == 0:
        return np.nan
    return tail_losses.mean()