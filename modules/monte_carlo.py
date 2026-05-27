# ============================================================
# modules/monte_carlo.py — Module 5
# Monte Carlo Simulation (Geometric Brownian Motion)
# ============================================================
# GBM models stock price paths as:
#   S(t+1) = S(t) × exp((μ − σ²/2)Δt + σ√Δt·Z)
# where Z ~ N(0,1).
# 1000+ paths give a distribution of possible future prices.
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"


def _run_gbm(S0: float, mu: float, sigma: float,
             T: int, n_sims: int, seed: int = 42) -> np.ndarray:
    """
    Simulate n_sims price paths over T days using GBM.
    Returns array of shape (T+1, n_sims).
    """
    rng      = np.random.default_rng(seed)
    dt       = 1 / 252                             # one trading day
    drift    = (mu - 0.5 * sigma ** 2) * dt
    diffusion= sigma * np.sqrt(dt)

    # Random shocks: shape (T, n_sims)
    Z        = rng.standard_normal((T, n_sims))
    log_ret  = drift + diffusion * Z               # daily log returns

    # Build paths: cumulative sum of log returns, then exponentiate
    paths    = np.zeros((T + 1, n_sims))
    paths[0] = S0
    paths[1:]= S0 * np.exp(np.cumsum(log_ret, axis=0))
    return paths


def render_monte_carlo(df, ticker, n_simulations, mc_horizon, **kwargs):
    """Render Module 5 — Monte Carlo Simulation."""

    st.subheader("🎲 Monte Carlo Price Simulation")
    st.markdown(
        f"**{n_simulations:,} simulation paths** over **{mc_horizon} trading days** "
        "using Geometric Brownian Motion (GBM). Adjust sliders in the sidebar."
    )

    close       = df["Close"].squeeze()
    log_returns = np.log(close / close.shift(1)).dropna()

    mu    = float(log_returns.mean()   * 252)   # annualised drift
    sigma = float(log_returns.std()    * np.sqrt(252))   # annualised vol
    S0    = float(close.iloc[-1])

    with st.spinner(f"Running {n_simulations:,} simulations …"):
        _t0 = time.time()
        paths = _run_gbm(S0, mu, sigma, mc_horizon, n_simulations)
        _elapsed = time.time() - _t0

    final_prices = paths[-1]     # terminal prices for all paths

    # ── Statistics ──
    expected_price = float(np.mean(final_prices))
    median_price   = float(np.median(final_prices))
    best_case      = float(np.percentile(final_prices, 95))
    worst_case     = float(np.percentile(final_prices, 5))
    prob_gain_10   = float(np.mean(final_prices > S0 * 1.10) * 100)
    prob_loss_10   = float(np.mean(final_prices < S0 * 0.90) * 100)

    # ── Sort paths for colouring ──
    sort_idx   = np.argsort(final_prices)
    bottom5_idx= sort_idx[:5]
    top5_idx   = sort_idx[-5:]

    # ============================================================
    # SIMULATION CHART
    # ============================================================
    fig = go.Figure()

    # Middle paths (blue, thin, semi-transparent)
    middle_idx = sort_idx[5:-5]
    step = max(1, len(middle_idx) // 200)   # plot at most 200 middle lines
    for i in middle_idx[::step]:
        fig.add_trace(go.Scatter(
            y=paths[:, i], mode="lines",
            line=dict(color="rgba(88,166,255,0.06)", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))

    # Bottom 5 (red)
    for j, i in enumerate(bottom5_idx):
        fig.add_trace(go.Scatter(
            y=paths[:, i], mode="lines",
            line=dict(color=RED, width=1.5),
            name="Bottom 5" if j == 0 else None,
            showlegend=(j == 0),
        ))

    # Top 5 (green)
    for j, i in enumerate(top5_idx):
        fig.add_trace(go.Scatter(
            y=paths[:, i], mode="lines",
            line=dict(color=GREEN, width=1.5),
            name="Top 5" if j == 0 else None,
            showlegend=(j == 0),
        ))

    # Mean path
    mean_path = paths.mean(axis=1)
    fig.add_trace(go.Scatter(
        y=mean_path, mode="lines",
        line=dict(color=YELLOW, width=2.5, dash="dash"),
        name="Mean Path",
    ))

    # Starting price line
    fig.add_hline(y=S0, line_dash="dot", line_color=ACCENT,
                  annotation_text=f"Current ₹{S0:,.2f}", annotation_font_color=ACCENT)

    fig.update_layout(
        title=f"Monte Carlo Simulation — {n_simulations:,} paths over {mc_horizon} days",
        xaxis_title="Trading Days",
        yaxis_title="Price (₹)",
        height=480,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TERMINAL PRICE DISTRIBUTION
    # ============================================================
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=final_prices, nbinsx=60,
        marker_color=ACCENT, opacity=0.75,
        name="Final Price Distribution",
    ))
    fig2.add_vline(x=S0,             line_dash="dash", line_color=YELLOW,
                   annotation_text="Current",     annotation_font_color=YELLOW)
    fig2.add_vline(x=expected_price, line_dash="dash", line_color=GREEN,
                   annotation_text="Expected",    annotation_font_color=GREEN)
    fig2.add_vline(x=worst_case,     line_dash="dot",  line_color=RED,
                   annotation_text="5th Pct",     annotation_font_color=RED)
    fig2.add_vline(x=best_case,      line_dash="dot",  line_color=PURPLE,
                   annotation_text="95th Pct",    annotation_font_color=PURPLE)

    fig2.update_layout(
        title="Distribution of Simulated Final Prices",
        xaxis_title="Final Price (₹)",
        yaxis_title="Frequency",
        height=300,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # METRICS
    # ============================================================
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Expected Price",  f"₹{expected_price:,.2f}",
              f"{(expected_price/S0-1)*100:+.1f}%")
    c2.metric("📊 Median Price",    f"₹{median_price:,.2f}",
              f"{(median_price/S0-1)*100:+.1f}%")
    c3.metric("🚀 Best Case (95%)", f"₹{best_case:,.2f}",
              f"{(best_case/S0-1)*100:+.1f}%")
    c4.metric("💀 Worst Case (5%)", f"₹{worst_case:,.2f}",
              f"{(worst_case/S0-1)*100:+.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("📈 P(gain > 10%)",   f"{prob_gain_10:.1f}%")
    c6.metric("📉 P(loss > 10%)",   f"{prob_loss_10:.1f}%")
    c7.metric("⚡ Ann. Drift μ",    f"{mu*100:.2f}%")
    c8.metric("📊 Ann. Vol σ",      f"{sigma*100:.2f}%")

    st.caption(f"⏱️ Computation time: {_elapsed:.2f}s for {n_simulations:,} paths over {mc_horizon} days")