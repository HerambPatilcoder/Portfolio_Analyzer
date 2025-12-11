import numpy as np
import pandas as pd


def simulate_portfolio_paths(
    daily_returns: pd.Series,
    initial_capital: float = 100000,
    num_paths: int = 5000,
    num_days: int = 252,
):
    """
    Simulate future portfolio value paths using a simple
    geometric Brownian motion approximation.

    daily_returns: historical portfolio daily returns (Series)
    initial_capital: starting portfolio value
    num_paths: number of simulated paths
    num_days: number of future trading days
    """
    # Estimate drift (mu) and volatility (sigma) from historical returns
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Pre-allocate array: rows = days, cols = paths
    paths = np.zeros((num_days + 1, num_paths))
    paths[0] = initial_capital

    # Generate random shocks
    # Normal(0, 1) scaled by sigma and drifted by mu
    dt = 1  # daily step
    rand = np.random.normal(0, 1, size=(num_days, num_paths))

    for t in range(1, num_days + 1):
        # Geometric Brownian Motion step
        paths[t] = paths[t - 1] * (1 + mu * dt + sigma * np.sqrt(dt) * rand[t - 1])

    # Create a DataFrame for convenience (index = day)
    index = range(num_days + 1)
    paths_df = pd.DataFrame(paths, index=index)

    # Summary statistics on final values
    final_values = paths_df.iloc[-1]
    summary = {
        "median_final": float(np.median(final_values)),
        "p5_final": float(np.percentile(final_values, 5)),
        "p95_final": float(np.percentile(final_values, 95)),
        "prob_loss": float(np.mean(final_values < initial_capital)),
    }

    return paths_df, summary