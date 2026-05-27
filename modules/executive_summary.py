# ============================================================
# modules/executive_summary.py — Module 1 (15 marks)
# Executive Summary Panel — All inline CSS, no class names
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time
from modules.ui_components import investment_signal, section_divider, render_html, COLORS

# ── Colour palette ──
C = COLORS
GREEN  = C["accent_green"]
RED    = C["accent_red"]
YELLOW = C["accent_amber"]
BLUE   = C["accent_blue"]
MUTED  = C["text_secondary"]
BG     = C["tertiary_dark"]
BORDER = C["border_subtle"]
TEXT   = C["text_primary"]


# ── Sparkline ──────────────────────────────────────────────
def _spark(series: pd.Series, color: str = BLUE) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=series.values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba(88,166,255,0.07)",
    ))
    fig.update_layout(
        height=55, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ── KPI card with fully inline styles ──────────────────────
def _kpi(label: str, value: str, sub: str, sub_color: str = MUTED,
         sparkline=None, col=None) -> None:
    target = col if col else st
    html = (
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;'
        f'padding:14px 16px;margin-bottom:6px;">'
        f'<div style="color:{MUTED};font-size:.74rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:.06em;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{BLUE};font-size:1.45rem;font-weight:700;line-height:1.2;">{value}</div>'
        f'<div style="color:{sub_color};font-size:.8rem;margin-top:5px;">{sub}</div>'
        f'</div>'
    )
    target.markdown(html, unsafe_allow_html=True)
    if sparkline is not None:
        target.plotly_chart(sparkline, use_container_width=True)


# ── Investment signal ───────────────────────────────────────
def _signal(mos: float, var95: float):
    if mos > 0.20 and var95 > -0.05:
        return "BUY",  GREEN, 80
    elif mos > 0.05:
        return "HOLD", YELLOW, 55
    else:
        return "SELL", RED, 35


def _risk(vol: float):
    if vol < 0.20:   return "Low",    GREEN
    elif vol < 0.35: return "Medium", YELLOW
    else:            return "High",   RED



@st.cache_data(ttl=3600, show_spinner=False)
def _get_eps_pe(tkr: str):
    """Module-level cached EPS/PE fetch."""
    for attempt in range(3):
        try:
            i = yf.Ticker(tkr).info
            return (i.get("trailingEps") or i.get("forwardEps") or 0,
                    i.get("trailingPE")  or i.get("forwardPE")  or 20)
        except Exception:
            time.sleep(2)
    return 0, 20

def render_executive_summary(df, ticker, all_close, wacc, terminal_g,
                              forecast_yrs, **kwargs):
    """Module 1 — Executive Summary Panel (15 marks)."""

    st.subheader("📋 Executive Summary")
    st.markdown(
        "Live KPI snapshot — price, forecasting, valuation, risk & portfolio metrics. "
        "Auto-refreshes on ticker/date change."
    )

    close       = df["Close"].squeeze()
    log_ret     = np.log(close / close.shift(1)).dropna()
    mu_d        = float(log_ret.mean())
    sigma_d     = float(log_ret.std())
    current_price = float(close.iloc[-1])
    prev_price    = float(close.iloc[-2])
    daily_ret     = (current_price - prev_price) / prev_price
    annual_vol    = sigma_d * np.sqrt(252)

    # ── Expected return (CAGR) ──
    n_yrs          = max(len(close) / 252, 0.01)
    total_ret      = (current_price / float(close.iloc[0])) - 1
    expected_return= (1 + total_ret) ** (1 / n_yrs) - 1

    # ── VaR 95% parametric ──
    var95 = mu_d - 1.645 * sigma_d

    # ── DCF quick estimate ──
    try:
        eps, pe = _get_eps_pe(ticker)
        dcf_value = (eps * (1 + expected_return) ** forecast_yrs * pe
                     / (1 + wacc) ** forecast_yrs) if eps and eps > 0 else current_price * 0.9
    except Exception:
        dcf_value = current_price * 0.9

    mos = (dcf_value - current_price) / max(dcf_value, 1)

    # ── Forecasted price (linear trend) ──
    x = np.arange(len(close))
    coeffs = np.polyfit(x, close.values, 1)
    forecasted_price = float(coeffs[0] * (len(close) + 90) + coeffs[1])

    # ── Portfolio stats ──
    port_ret_df = all_close.pct_change().dropna()
    if not port_ret_df.empty and port_ret_df.shape[1] > 0:
        w            = np.ones(port_ret_df.shape[1]) / port_ret_df.shape[1]
        daily_port   = (port_ret_df * w).sum(axis=1)
        port_return  = float(daily_port.mean() * 252)
        port_risk    = float(daily_port.std()  * np.sqrt(252))
        sharpe       = port_return / port_risk if port_risk > 0 else 0.0
    else:
        port_return = port_risk = sharpe = 0.0

    prob_default = max(0.01, min(0.99, 1 / (1 + np.exp(2 * (sharpe - 0.5)))))

    sig_text, sig_color, confidence = _signal(mos, var95)
    risk_label, risk_color          = _risk(annual_vol)

    # ===========================================================
    # 1. INVESTMENT SIGNAL BOX
    # ===========================================================
    explanation = (
        f"DCF Intrinsic: ₹{dcf_value:,.0f} (MoS: {mos*100:.1f}%) "
        f"| VaR 95%: {var95*100:.2f}% | Risk: {risk_label}"
    )
    investment_signal(sig_text, confidence, risk_label, explanation)

    section_divider()

    # ===========================================================
    # 2. TOP KPI ROW — Price trio with sparklines (1a requirement)
    # ===========================================================
    st.markdown(
        f'<p style="color:{MUTED};font-size:.82rem;margin-bottom:8px;">'
        f'📡 Live data · {ticker} · auto-updates on ticker/date change</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    last90 = close.iloc[-90:]

    with c1:
        _kpi("💹 Current Price",
             f"₹{current_price:,.2f}",
             f"{'↑' if daily_ret >= 0 else '↓'} {abs(daily_ret)*100:.2f}% today",
             GREEN if daily_ret >= 0 else RED)
        st.plotly_chart(_spark(last90, BLUE), use_container_width=True)

    with c2:
        _kpi("🔮 Forecasted Price (90D)",
             f"₹{forecasted_price:,.2f}",
             f"{'↑' if forecasted_price > current_price else '↓'} "
             f"{abs(forecasted_price/current_price - 1)*100:.2f}% vs now",
             GREEN if forecasted_price > current_price else RED)
        trend_series = pd.Series(
            [float(coeffs[0]*i + coeffs[1]) for i in range(len(close)-90, len(close)+90)]
        )
        st.plotly_chart(_spark(trend_series, GREEN), use_container_width=True)

    with c3:
        _kpi("📐 DCF Intrinsic Value",
             f"₹{dcf_value:,.2f}",
             f"MoS {mos*100:.1f}% · {'✓ Undervalued' if mos > 0 else '✗ Overvalued'}",
             GREEN if mos > 0 else RED)
        rolling_iv = close.rolling(20).mean()
        st.plotly_chart(_spark(rolling_iv.iloc[-90:], YELLOW), use_container_width=True)

    section_divider()

    # ===========================================================
    # 3. KPI METRICS GRID (1b requirement — 9 live metrics)
    # ===========================================================
    st.markdown(
        f'<div style="color:{TEXT};font-size:1rem;font-weight:700;margin-bottom:12px;">'
        f'📊 Nine Live Risk & Performance Metrics</div>',
        unsafe_allow_html=True,
    )

    metrics = [
        ("📈 Expected Return (1Y)", f"{expected_return*100:+.1f}%",
         "CAGR Projection", GREEN if expected_return > 0 else RED),
        ("🗂️ Portfolio Return (1Y)", f"{port_return*100:+.1f}%",
         "Equal-weight 5-stock", GREEN if port_return > 0 else RED),
        ("📉 Portfolio Risk (Vol)", f"{port_risk*100:.1f}%",
         "Annualised Std Dev", YELLOW),
        ("⚠️ VaR 95% (1-Day)", f"{var95*100:.3f}%",
         "Parametric Normal", RED if var95 < -0.05 else GREEN),
        ("🏦 Prob. of Default", f"{prob_default*100:.2f}%",
         "Logistic proxy", GREEN if prob_default < 0.1 else (YELLOW if prob_default < 0.3 else RED)),
        ("⚡ Sharpe Ratio (1Y)", f"{sharpe:.3f}",
         "Return / Risk", GREEN if sharpe > 1 else (YELLOW if sharpe > 0 else RED)),
        ("📊 Annual Volatility", f"{annual_vol*100:.2f}%",
         risk_label + " regime", risk_color),
        ("🎯 Margin of Safety", f"{mos*100:.1f}%",
         "DCF vs Market Price", GREEN if mos > 0.20 else (YELLOW if mos > 0 else RED)),
        ("📆 Trading Days", f"{len(close):,}",
         "Data points fetched", MUTED),
    ]

    rows = [metrics[i:i+3] for i in range(0, len(metrics), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (label, value, sub, color) in zip(cols, row):
            col.markdown(
                f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;'
                f'padding:14px 16px;margin-bottom:8px;">'
                f'<div style="color:{MUTED};font-size:.74rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">{label}</div>'
                f'<div style="color:{BLUE};font-size:1.35rem;font-weight:700;">{value}</div>'
                f'<div style="color:{color};font-size:.79rem;margin-top:4px;">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    section_divider()

    # ===========================================================
    # 4. FULL PRICE CHART with MA50 / MA200
    # ===========================================================
    st.markdown(
        f'<div style="color:{TEXT};font-size:1rem;font-weight:700;margin-bottom:8px;">'
        f'📈 Price History — {ticker}</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close",
                             line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(),
                             name="MA 50", line=dict(color=YELLOW, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(200).mean(),
                             name="MA 200", line=dict(color=RED, width=1, dash="dot")))
    fig.update_layout(
        height=380, template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis_title="Date", yaxis_title="Price (₹)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # 5. ROLLING METRICS SPARKLINES
    # ===========================================================
    section_divider()
    st.markdown(
        f'<div style="color:{TEXT};font-size:1rem;font-weight:700;margin-bottom:8px;">'
        f'📉 Rolling 20-Day Metrics (Last 90 Days)</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    roll_vol = log_ret.rolling(20).std() * np.sqrt(252) * 100
    roll_ret = close.pct_change().rolling(20).mean() * 252 * 100
    roll_sr  = roll_ret / (roll_vol + 1e-9)

    with s1:
        st.caption("Rolling Volatility (%)")
        st.plotly_chart(_spark(roll_vol.iloc[-90:], YELLOW), use_container_width=True)
    with s2:
        st.caption("Rolling Annualised Return (%)")
        st.plotly_chart(_spark(roll_ret.iloc[-90:], GREEN), use_container_width=True)
    with s3:
        st.caption("Rolling Sharpe Ratio")
        st.plotly_chart(_spark(roll_sr.iloc[-90:], BLUE), use_container_width=True)