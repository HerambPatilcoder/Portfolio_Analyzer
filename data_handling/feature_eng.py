import pandas as pd
import numpy as np


def compute_daily_returns(price_df):
    """
    price_df: DataFrame of adjusted close prices (dates × tickers)
    """
    returns = price_df.pct_change().dropna(how="all")
    return returns


def portfolio_returns(returns_df, portfolio_df):
    """
    returns_df: DataFrame of daily returns (dates × tickers)
    portfolio_df: DataFrame with columns [ticker, weight]

    returns: Series of portfolio daily returns
    """
    weights = portfolio_df.set_index("ticker")["weight"]
    # align weights with columns in returns_df
    weights = weights.reindex(returns_df.columns).fillna(0.0)
    port_ret = returns_df.dot(weights)
    return port_ret


def annualized_return(daily_returns):
    return daily_returns.mean() * 252


def annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

