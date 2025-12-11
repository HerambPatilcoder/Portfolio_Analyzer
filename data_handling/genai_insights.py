import os
from typing import Dict, Any

from groq import Groq


def _get_client():
    """
    Returns a Groq client.
    Requires GROQ_API_KEY to be set in environment or Streamlit secrets.
    """
    api_key = os.get_env("GROQ_API_KEY")

    # Optional: allow Streamlit secrets
    try:
        import streamlit as st
        if not api_key and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Set it in your environment or Streamlit secrets."
        )

    return Groq(api_key=api_key)


def generate_portfolio_insights(
    portfolio_df,
    risk_metrics: Dict[str, float],
    optimization: Dict[str, Any] | None = None,
    simulation: Dict[str, float] | None = None,
    model: str = "llama-3.1-8b-instant",  # ✅ fast & free-tier friendly
) -> str:
    """
    Uses Groq-hosted LLMs (LLaMA / Mixtral) to generate
    an investor-friendly portfolio analysis.
    """
    client = _get_client()

    # -------- Build plain-text summary --------

    lines = []
    lines.append("Portfolio Composition:")
    for _, row in portfolio_df.iterrows():
        lines.append(
            f"- {row['ticker']}: {row['shares']} shares, "
            f"weight {row['weight']:.2%}"
        )

    lines.append("\nKey Risk & Performance Metrics (last 10 years):")
    lines.append(f"- Annualized return: {risk_metrics.get('ann_return', 0):.2%}")
    lines.append(f"- Annualized volatility: {risk_metrics.get('ann_vol', 0):.2%}")
    lines.append(f"- Sharpe ratio: {risk_metrics.get('sharpe', float('nan')):.2f}")
    lines.append(f"- Maximum drawdown: {risk_metrics.get('max_dd', 0):.2%}")
    lines.append(f"- Daily VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
    lines.append(f"- Daily CVaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")

    if optimization is not None:
        ms = optimization.get("max_sharpe")
        mv = optimization.get("min_vol")

        lines.append("\nOptimization Results (PyPortfolioOpt):")
        if ms:
            lines.append(
                f"- Max Sharpe portfolio: "
                f"return {ms['ret']:.2%}, vol {ms['vol']:.2%}, sharpe {ms['sharpe']:.2f}"
            )
        if mv:
            lines.append(
                f"- Min Volatility portfolio: "
                f"return {mv['ret']:.2%}, vol {mv['vol']:.2%}, sharpe {mv['sharpe']:.2f}"
            )

    if simulation is not None:
        lines.append("\nMonte Carlo Simulation Summary:")
        lines.append(f"- Median final value: {simulation.get('median_final', 0):,.0f}")
        lines.append(f"- 5th percentile final value: {simulation.get('p5_final', 0):,.0f}")
        lines.append(f"- 95th percentile final value: {simulation.get('p95_final', 0):,.0f}")
        lines.append(
            f"- Probability of loss: {simulation.get('prob_loss', 0):.2%}"
        )

    stats_text = "\n".join(lines)

    # -------- LLM Prompt --------

    system_msg = (
    "You are a friendly financial guide explaining portfolio performance to a "
    "normal retail investor with no finance background. "
    "Use very simple language. "
    "Avoid technical finance terms like Sharpe, volatility, VaR, CVaR, optimization. "
    "If such concepts are present, explain them in plain English instead. "
    "Do not give direct buy/sell advice. "
    "Your goal is to help the user understand what happened and what it means for them."
    )

    user_msg = (
    "Here is the portfolio information and results in raw form:\n\n"
    f"{stats_text}\n\n"
    "Now write a clear, simple explanation that includes:\n"
    "1. How the portfolio has performed overall (good or risky?)\n"
    "2. Whether the portfolio is too dependent on a few stocks\n"
    "3. What kind of losses the user should be mentally prepared for\n"
    "4. How the safer and balanced versions compare to the current one\n"
    "5. What the future simulations suggest in simple money terms\n"
    "6. A final section titled 'What This Means for You' written as bullet points\n\n"
    "Rules:\n"
    "- Use only simple English\n"
    "- Use short sentences\n"
    "- Avoid finance jargon\n"
    "- Do not mention formulas or model names\n"
    "- Do not say you are an AI\n"
    "- Do not give personalized investment advice\n"
    "- Do not promise profits"
    )

    # -------- Groq API Call --------

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=700,
    )

    return completion.choices[0].message.content.strip()
