import pandas as pd
import yfinance as yf
from datetime import date, timedelta

def build_portfolio(tickers, shares):
    """
    tickers: list of ticker symbols (strings)
    shares: list of number of shares (ints/floats)

    returns: DataFrame with columns [ticker, shares, weight]
    """
    if len(tickers) != len(shares):
        raise ValueError("Length of tickers and shares must match.")

    df = pd.DataFrame({
        "ticker": [t.strip().upper() for t in tickers],
        "shares": shares
    })

    if (df["shares"] <= 0).any():
        raise ValueError("All share quantities must be positive.")

    total_shares = df["shares"].sum()
    df["weight"] = df["shares"] / total_shares

    return df

def fetch_stock_data(tickers, start, end):
    """
    Generic function to fetch adjusted close prices for given tickers.
    """
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    data = df["Close"].copy()
    data.dropna(how="all", inplace=True)
    return data


def fetch_10y_history_for_portfolio(portfolio_df):
    """
    portfolio_df: DataFrame with at least a 'ticker' column.

    Fetches last 10 years of adjusted close prices for unique tickers.
    """
    unique_tickers = portfolio_df["ticker"].unique().tolist()

    end = date.today()
    start = end - timedelta(days=365 * 10)  # approx 10 years

    price_df = fetch_stock_data(unique_tickers, start=start, end=end)

    return price_df, start, end

