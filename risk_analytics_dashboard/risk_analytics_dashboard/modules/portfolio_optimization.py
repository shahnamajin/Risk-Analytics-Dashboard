# ============================================================
# modules/portfolio_optimization.py — Module 8
# Mean-Variance Portfolio Optimization (Efficient Frontier)
# ============================================================
# 5000 random portfolios are simulated across 5 stocks +
# Gold (GLD) + Indian Bond proxy (CRISIL).
# The maximum Sharpe Ratio portfolio is highlighted.
# PyPortfolioOpt is used for the exact optimal weights.
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"

ALL_ASSETS = {
    "INFY.NS":     "Infosys",
    "RELIANCE.NS": "Reliance",
    "ITC.NS":      "ITC",
    "HDFCBANK.NS": "HDFC Bank",
    "ABB.NS":      "ABB India",
    "GLD":         "Gold ETF",
    "^NSEI":       "Nifty 50",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_assets(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    raw   = yf.download(list(tickers), start=start, end=end,
                        auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw.rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw
    return close.dropna(how="all")


def render_portfolio_optimization(all_close, start_date, end_date, **kwargs):
    """Render Module 8 — Portfolio Optimization."""

    st.subheader("🗂️ Portfolio Optimization — Efficient Frontier")
    st.markdown(
        "5,000 random portfolios plotted as a Markowitz Efficient Frontier. "
        "The **Maximum Sharpe Ratio** portfolio is highlighted."
    )

    # ── Asset toggle ──
    available = list(ALL_ASSETS.keys())
    selected_assets = st.multiselect(
        "Select Assets",
        options=available,
        default=available[:5],
        format_func=lambda x: f"{ALL_ASSETS[x]} ({x})",
    )
    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets.")
        return

    with st.spinner("Fetching asset prices …"):
        prices = _fetch_assets(tuple(selected_assets), start_date, end_date)

    # Keep only columns we have data for
    prices.dropna(axis=1, how="all", inplace=True)
    valid   = [t for t in selected_assets if t in prices.columns]
    prices  = prices[valid].dropna()

    if prices.shape[1] < 2:
        st.error("Not enough valid assets after downloading. Try different assets.")
        return

    n_assets = len(valid)
    log_rets = np.log(prices / prices.shift(1)).dropna()
    mu_daily = log_rets.mean()
    cov_daily= log_rets.cov()

    ann_mu   = mu_daily * 252
    ann_cov  = cov_daily * 252

    # ── 5000 Random Portfolios ──
    N = 5000
    rng = np.random.default_rng(99)

    port_returns = np.zeros(N)
    port_vols    = np.zeros(N)
    port_sharpe  = np.zeros(N)
    port_weights = np.zeros((N, n_assets))

    rf = 0.065    # risk-free rate (approx India 10Y Gsec)

    for i in range(N):
        w = rng.random(n_assets)
        w /= w.sum()
        port_weights[i] = w
        r = float(w @ ann_mu.values)
        v = float(np.sqrt(w @ ann_cov.values @ w))
        port_returns[i] = r
        port_vols[i]    = v
        port_sharpe[i]  = (r - rf) / v if v > 0 else 0

    max_sr_idx = np.argmax(port_sharpe)
    opt_w      = port_weights[max_sr_idx]
    opt_ret    = port_returns[max_sr_idx]
    opt_vol    = port_vols[max_sr_idx]
    opt_sharpe = port_sharpe[max_sr_idx]

    min_vol_idx = np.argmin(port_vols)

    # ── Efficient Frontier chart ──
    fig = go.Figure()

    # All portfolios (colour = Sharpe)
    fig.add_trace(go.Scatter(
        x=port_vols * 100,
        y=port_returns * 100,
        mode="markers",
        marker=dict(
            color=port_sharpe,
            colorscale="Viridis",
            size=3,
            opacity=0.6,
            colorbar=dict(title="Sharpe"),
        ),
        name="Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%",
    ))

    # Max Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode="markers",
        marker=dict(color=GREEN, size=16, symbol="star"),
        name=f"Max Sharpe ({opt_sharpe:.2f})",
    ))

    # Min Vol portfolio
    fig.add_trace(go.Scatter(
        x=[port_vols[min_vol_idx] * 100],
        y=[port_returns[min_vol_idx] * 100],
        mode="markers",
        marker=dict(color=YELLOW, size=14, symbol="diamond"),
        name="Min Volatility",
    ))

    fig.update_layout(
        title="Efficient Frontier — 5,000 Random Portfolios",
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        height=480,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Allocation pie chart ──
    labels = [ALL_ASSETS.get(t, t) for t in valid]
    pie_fig = go.Figure(go.Pie(
        labels=labels,
        values=opt_w * 100,
        hole=0.4,
        marker=dict(colors=[
            ACCENT, GREEN, YELLOW, RED, PURPLE, "#ff7b72", "#79c0ff"
        ][:n_assets]),
        textinfo="label+percent",
    ))
    pie_fig.update_layout(
        title="Max Sharpe Portfolio Allocation",
        height=360,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # ── Allocation table ──
    alloc_df = pd.DataFrame({
        "Asset":      [f"{ALL_ASSETS.get(t, t)} ({t})" for t in valid],
        "Weight (%)": [f"{w*100:.2f}%" for w in opt_w],
    })
    st.table(alloc_df)

    # ── Metrics ──
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Max Sharpe Ratio",   f"{opt_sharpe:.3f}")
    c2.metric("📈 Portfolio Return",   f"{opt_ret*100:.2f}%")
    c3.metric("📉 Portfolio Risk",     f"{opt_vol*100:.2f}%")
