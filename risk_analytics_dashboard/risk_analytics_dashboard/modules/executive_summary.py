# ============================================================
# modules/executive_summary.py — Module 1
# Executive Summary Panel with KPI Cards & Investment Signal
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Reusable colour helper ──
CARD_BG   = "#1c2128"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"


def _sparkline(series: pd.Series, color: str = ACCENT) -> go.Figure:
    """Return a tiny Plotly line chart (no axes, no margin) for a KPI card."""
    fig = go.Figure(
        go.Scatter(
            y=series.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba(88,166,255,0.08)",
        )
    )
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _investment_signal(margin_of_safety: float, var_95: float):
    """
    Decision logic:
      BUY  → MoS > 20 %  AND  VaR > -5 %
      HOLD → MoS 5–20 %
      SELL → otherwise
    """
    if margin_of_safety > 0.20 and var_95 > -0.05:
        return "🟢 BUY", GREEN
    elif margin_of_safety > 0.05:
        return "🟡 HOLD", YELLOW
    else:
        return "🔴 SELL", RED


def _risk_badge(annual_vol: float):
    """Classify annual volatility into Low / Medium / High."""
    if annual_vol < 0.20:
        return "🟢 Low Risk",    GREEN
    elif annual_vol < 0.35:
        return "🟡 Medium Risk", YELLOW
    else:
        return "🔴 High Risk",   RED


def render_executive_summary(df, ticker, all_close, wacc, terminal_g,
                              forecast_yrs, **kwargs):
    """
    Render Module 1 — Executive Summary.

    Parameters
    ----------
    df          : OHLCV DataFrame for the selected ticker
    ticker      : str, e.g. 'INFY.NS'
    all_close   : DataFrame of Close prices for all tickers (portfolio use)
    wacc        : float  (decimal, e.g. 0.10)
    terminal_g  : float  (decimal, e.g. 0.03)
    forecast_yrs: int
    """
    st.subheader("📋 Executive Summary")
    st.markdown(
        "High-level KPI snapshot combining price data, valuation, risk, and portfolio analytics."
    )

    close = df["Close"].squeeze()          # 1-D Series

    # ── Basic price metrics ──
    current_price   = float(close.iloc[-1])
    prev_price      = float(close.iloc[-2])
    daily_ret       = (current_price - prev_price) / prev_price
    log_returns     = np.log(close / close.shift(1)).dropna()
    annual_vol      = float(log_returns.std() * np.sqrt(252))

    # ── Expected 1-year return (CAGR proxy) ──
    n_years         = len(close) / 252
    total_return    = (current_price / float(close.iloc[0])) - 1
    expected_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # ── Simple DCF intrinsic value (fast estimate; full DCF in Module 4) ──
    # EPS-based shortcut: Price × (1 + expected_return)^5 discounted at WACC
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        eps  = info.get("trailingEps", None) or info.get("forwardEps", None)
        pe   = info.get("trailingPE", None)  or info.get("forwardPE", None) or 20
        if eps and eps > 0:
            projected_eps = eps * (1 + expected_return) ** forecast_yrs
            dcf_value     = projected_eps * pe / (1 + wacc) ** forecast_yrs
        else:
            dcf_value = current_price * 0.9          # fallback
    except Exception:
        dcf_value = current_price * 0.9

    margin_of_safety = (dcf_value - current_price) / dcf_value

    # ── Forecast price (simple linear extrapolation; ARIMA in Module 2) ──
    from numpy.polynomial.polynomial import polyfit
    x      = np.arange(len(close))
    coeffs = polyfit(x, close.values, 1)
    # slope × 90 days ahead
    forecasted_price = float(coeffs[0] + coeffs[1] * (len(close) + 90))

    # ── VaR 95 % (parametric) ──
    mu    = float(log_returns.mean())
    sigma = float(log_returns.std())
    var95 = float(mu - 1.645 * sigma)        # 1-day log-return VaR

    # ── Portfolio return & risk (equal-weight across available tickers) ──
    port_returns = all_close.pct_change().dropna()
    if not port_returns.empty:
        weights       = np.ones(port_returns.shape[1]) / port_returns.shape[1]
        port_ret_daily = (port_returns * weights).sum(axis=1)
        port_return   = float(port_ret_daily.mean() * 252)
        port_risk     = float(port_ret_daily.std()  * np.sqrt(252))
        sharpe        = port_return / port_risk if port_risk > 0 else 0.0
    else:
        port_return, port_risk, sharpe = 0.0, 0.0, 0.0

    # ── Probability of Default proxy (Altman Z-score flavour; full model in Module 7) ──
    prob_default = max(0.01, min(0.99, 1 / (1 + np.exp(2 * (sharpe - 0.5)))))

    # ── Signal & badge ──
    signal_label, signal_color = _investment_signal(margin_of_safety, var95)
    badge_label,  badge_color  = _risk_badge(annual_vol)

    # ===========================================================
    # LAYOUT: Signal banner → KPI cards → sparklines
    # ===========================================================

    # Signal banner
    st.markdown(
        f"""
        <div style='background:{signal_color}22; border:1px solid {signal_color};
                    border-radius:8px; padding:12px 20px; margin-bottom:16px;
                    display:flex; align-items:center; gap:16px;'>
            <span style='font-size:28px;'>{signal_label}</span>
            <span style='color:#e6edf3; font-size:14px;'>
                Investment signal based on DCF Margin of Safety
                (<b>{margin_of_safety*100:.1f}%</b>) and 1-day VaR
                (<b>{var95*100:.2f}%</b>)
            </span>
            <span style='margin-left:auto; background:{badge_color}33;
                         border:1px solid {badge_color}; border-radius:20px;
                         padding:4px 12px; color:{badge_color}; font-size:13px;'>
                {badge_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Row 1: price KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💹 Current Price",      f"₹{current_price:,.2f}",
              f"{daily_ret*100:+.2f}% today")
    c2.metric("🔮 Forecasted Price",   f"₹{forecasted_price:,.2f}",
              f"{(forecasted_price/current_price-1)*100:+.2f}% vs now")
    c3.metric("📐 DCF Intrinsic Value", f"₹{dcf_value:,.2f}",
              f"MoS {margin_of_safety*100:.1f}%")
    c4.metric("📈 Expected Return (1Y)", f"{expected_return*100:.1f}%")

    # ── Row 2: portfolio & risk KPIs ──
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🗂️ Portfolio Return",    f"{port_return*100:.1f}%")
    c6.metric("📉 Portfolio Risk",      f"{port_risk*100:.1f}%")
    c7.metric("⚠️ VaR 95% (1-day)",    f"{var95*100:.2f}%")
    c8.metric("🏦 Prob. of Default",    f"{prob_default*100:.1f}%")

    # ── Row 3: derived KPIs ──
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("⚡ Sharpe Ratio",        f"{sharpe:.2f}")
    c10.metric("📊 Annual Volatility",  f"{annual_vol*100:.1f}%")
    c11.metric("🎯 Margin of Safety",   f"{margin_of_safety*100:.1f}%",
               "Undervalued" if margin_of_safety > 0 else "Overvalued")
    c12.metric("📆 Data Points",        f"{len(close):,}")

    st.markdown("---")

    # ── Sparkline panel ──
    st.markdown("#### 📉 Price Sparklines (Last 90 Days)")
    sc1, sc2, sc3 = st.columns(3)

    last90 = close.iloc[-90:]
    with sc1:
        st.markdown("**Close Price**")
        st.plotly_chart(_sparkline(last90, ACCENT), use_container_width=True)

    rolling_vol = log_returns.rolling(20).std() * np.sqrt(252) * 100
    with sc2:
        st.markdown("**Rolling 20-Day Volatility (%)**")
        st.plotly_chart(_sparkline(rolling_vol.iloc[-90:], YELLOW), use_container_width=True)

    rolling_ret = close.pct_change().rolling(20).mean() * 252 * 100
    with sc3:
        st.markdown("**Rolling 20-Day Annualised Return (%)**")
        st.plotly_chart(_sparkline(rolling_ret.iloc[-90:], GREEN), use_container_width=True)

    # ── Full price chart ──
    st.markdown("---")
    st.markdown("#### 📈 Full Price History with 50-Day & 200-Day MA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, name="Close",
        line=dict(color=ACCENT, width=1.5),
    ))
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    fig.add_trace(go.Scatter(x=ma50.index,  y=ma50.values,  name="MA 50",
                             line=dict(color=YELLOW, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA 200",
                             line=dict(color=RED,    width=1, dash="dot")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", y=1.02, x=0),
        height=380,
        margin=dict(l=50, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
