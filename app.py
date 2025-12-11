import streamlit as st
import pandas as pd

from data_handling.portfolio_utils import build_portfolio,fetch_10y_history_for_portfolio
from data_handling.feature_eng import compute_daily_returns, portfolio_returns, annualized_return, annualized_volatility
from data_handling.risk_metrics import sharpe_ratio, compute_drawdown, value_at_risk, conditional_var
from data_handling.monte_carlo import simulate_portfolio_paths
from data_handling.optimization import run_optimizations
from data_handling.genai_insights import generate_portfolio_insights

st.sidebar.header("Simulation Settings")

initial_capital = st.sidebar.number_input(
    "Initial Portfolio Value",
    min_value=1000.0,
    value=100000.0,
    step=1000.0,
)

num_paths = st.sidebar.number_input(
    "Number of Simulation Paths",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100,
)

num_days = st.sidebar.number_input(
    "Days to Simulate",
    min_value=30,
    max_value=252 * 5,
    value=252,
    step=30,
)

st.sidebar.header("AI Analysis Settings")

enable_genai = st.sidebar.checkbox("Generate AI Portfolio Analysis", value=True)

model_name = st.sidebar.text_input(
    "Groq Model Name",
    value="llama-3.1-8b-instant",
    help="Options: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b"
)

st.set_page_config(page_title="AI Portfolio Analyzer", layout="wide")


st.title("📊 AI-Powered Portfolio Performance & Risk Analyzer")

st.markdown(
    """
Enter your stock portfolio below.  
Provide the **ticker symbols** and **number of shares** you own.  
The app will:
- Compute the weights of each stock based on shares
- Fetch the **last 10 years** of price data
- Compute basic performance statistics
"""
)

# --------- USER INPUT SECTION ---------

st.subheader("1️⃣ Portfolio Input")

col1, col2 = st.columns(2)

with col1:
    tickers_input = st.text_input(
        "Enter tickers (comma-separated)",
        value="AAPL,MSFT,TSLA"
    )

with col2:
    shares_input = st.text_input(
        "Enter corresponding number of shares (comma-separated)",
        value="10,5,3"
    )

run_button = st.button("Run Portfolio Analysis")


def parse_list_from_input(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list_from_input(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


if run_button:
    try:
        # Parse user input
        tickers = parse_list_from_input(tickers_input)
        shares = parse_float_list_from_input(shares_input)

        if not tickers or not shares:
            st.error("Please enter at least one ticker and share quantity.")
        else:
            # Build portfolio DataFrame with weights
            portfolio_df = build_portfolio(tickers, shares)

            st.subheader("📁 Your Portfolio")
            st.dataframe(portfolio_df, width='stretch')

            # Fetch 10-year price history
            with st.spinner("Fetching last 10 years of price data..."):
                price_df, start_date, end_date = fetch_10y_history_for_portfolio(portfolio_df)

            if price_df.empty:
                st.error("No price data found for the given tickers and date range.")
            else:
                st.success(f"Fetched data from {start_date} to {end_date}.")
                st.subheader("📈 Price Data Preview")
                st.dataframe(price_df.tail(), width='stretch')

                # Compute returns
                returns_df = compute_daily_returns(price_df)
                port_ret = portfolio_returns(returns_df, portfolio_df)

                # Basic stats
                ann_ret = annualized_return(port_ret)
                ann_vol = annualized_volatility(port_ret)
                
                # ---- Risk metrics ----
                rf_rate = 0.0661  # example: 2% annual risk-free rate, you can make this a UI input later
                sharpe = sharpe_ratio(port_ret, risk_free_rate=rf_rate)
                drawdown_series, max_dd = compute_drawdown(port_ret)
                var_95 = value_at_risk(port_ret, confidence=0.95)
                cvar_95 = conditional_var(port_ret, confidence=0.95)

                st.subheader("🧮 Portfolio Optimization (PyPortfolioOpt)")

                with st.spinner("Running mean-variance optimization..."):
                    opt_results = run_optimizations(price_df, risk_free_rate=rf_rate)

                max_sharpe_res = opt_results["max_sharpe"]
                min_vol_res = opt_results["min_vol"]

                # ---- Show performance comparison ----
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    st.metric(
                        "Current Ann. Return",
                        f"{ann_ret * 100:.2f}%",
                        help="Based on your current weights"
                   )
                with perf_col2:
                    st.metric(
                        "Max Sharpe Ann. Return",
                        f"{max_sharpe_res['ret'] * 100:.2f}%",
                    )
                with perf_col3:
                    st.metric(
                        "Min Vol Ann. Return",
                        f"{min_vol_res['ret'] * 100:.2f}%",
                    )
                st.caption(
                    "Max Sharpe: Optimized for highest risk-adjusted return."
                    "Min Vol: Optimized for lowest volatility."
                )    

                # ---- Weight comparison table ----
                tickers = list(price_df.columns)
                current_w = (
                    portfolio_df.set_index("ticker")["weight"]
                    .reindex(tickers)
                    .fillna(0.0)
                )
                max_sharpe_w = (
                    pd.Series(max_sharpe_res["weights"])
                    .reindex(tickers)
                    .fillna(0.0)
                )
                min_vol_w = (
                    pd.Series(min_vol_res["weights"])
                    .reindex(tickers)
                    .fillna(0.0)
                )
                weights_df = pd.DataFrame({
                    "Current": current_w,
                    "Max Sharpe": max_sharpe_w,
                    "Min Vol": min_vol_w,
                })

                st.subheader("⚖️ Weight Allocation Comparison")
                st.dataframe(weights_df.round(4), width="stretch")

                st.subheader("🎲 Monte Carlo Simulation (Future Portfolio Values)")

                with st.spinner("Running Monte Carlo simulation..."):
                    paths_df, sim_summary = simulate_portfolio_paths(
                        port_ret,
                        initial_capital=initial_capital,
                        num_paths=int(num_paths),
                        num_days=int(num_days),
                    )

                # Show summary metrics
                sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
                with sim_col1:
                    st.metric("Median Final Value", f"{sim_summary['median_final']:,.0f}")
                with sim_col2:
                    st.metric("5th Percentile", f"{sim_summary['p5_final']:,.0f}")
                with sim_col3:
                    st.metric("95th Percentile", f"{sim_summary['p95_final']:,.0f}")
                with sim_col4:
                    st.metric("Prob. of Loss", f"{sim_summary['prob_loss'] * 100:.1f}%")

                # Plot a subset of paths to avoid overplotting
                st.caption("Showing a subset of simulated paths for clarity.")
                num_paths_to_plot = min(50, paths_df.shape[1])
                st.line_chart(paths_df.iloc[:, :num_paths_to_plot], width="stretch")

                st.subheader("📌 Portfolio Statistics (Last 10 Years)")

                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Annualized Return", f"{ann_ret * 100:.2f}%")
                with stat_col2:
                    st.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
                with stat_col3:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe == sharpe else "N/A")

                stat_col4, stat_col5 = st.columns(2)
                with stat_col4:
                    st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
                with stat_col5:
                    st.metric("VaR 95% (Daily)", f"{var_95 * 100:.2f}%")
                    # Optionally show CVaR below
                    st.caption(f"CVaR 95% (Daily): {cvar_95 * 100:.2f}%")

                # --------- Build Summary Dicts for GenAI ---------

                risk_metrics = {
                    "ann_return": float(ann_ret),
                    "ann_vol": float(ann_vol),
                    "sharpe": float(sharpe) if sharpe == sharpe else 0.0,
                    "max_dd": float(max_dd),
                    "var_95": float(var_95),
                    "cvar_95": float(cvar_95),
                }

                optimization_summary = {
                    "max_sharpe": opt_results.get("max_sharpe"),
                    "min_vol": opt_results.get("min_vol"),
                }

                simulation_summary = sim_summary if "sim_summary" in locals() else None


                # --------- GenAI Section (Groq) ---------
                if enable_genai:
                    st.subheader("🧠 AI-Generated Portfolio Analysis")

                    try:
                        with st.spinner("Generating AI analysis using Groq..."):
                            ai_text = generate_portfolio_insights(
                                portfolio_df=portfolio_df,
                                risk_metrics=risk_metrics,
                                optimization=optimization_summary,
                                simulation=simulation_summary,
                                model=model_name,
                            )

                        st.markdown(ai_text)

                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
                        st.info(
                            "Check that GROQ_API_KEY is correctly set in .streamlit/secrets.toml"
                        )

    except Exception as e:
        st.error(f"Error: {e}")
    
    st.subheader("📈 Portfolio Cumulative Returns")
    cum_portfolio = (1 + port_ret).cumprod()
    st.line_chart(cum_portfolio, width="stretch")