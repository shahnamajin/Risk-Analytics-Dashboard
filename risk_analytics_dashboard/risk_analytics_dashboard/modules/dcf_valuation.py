# ============================================================
# modules/dcf_valuation.py — Module 4
# Discounted Cash Flow (DCF) Valuation
# ============================================================
# DCF estimates a stock's intrinsic value by:
#   1. Projecting Free Cash Flow (FCF) into the future
#   2. Discounting each year at WACC
#   3. Adding a Terminal Value (perpetuity growth model)
#   4. Dividing Enterprise Value by shares outstanding
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


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a yfinance value (may be None or ndarray) to float."""
    try:
        if val is None:
            return default
        v = float(val) if not hasattr(val, "__iter__") else float(val.iloc[0])
        return v if not np.isnan(v) else default
    except Exception:
        return default


def render_dcf_valuation(df, ticker, wacc, terminal_g, forecast_yrs, **kwargs):
    """Render Module 4 — DCF Valuation."""

    st.subheader("💰 DCF Intrinsic Value Estimation")
    st.markdown(
        "Discounted Cash Flow analysis using real Free Cash Flow data from yfinance. "
        "Adjust **WACC**, **Terminal Growth**, and **Forecast Years** in the sidebar."
    )

    # ── Fetch fundamental data ──
    with st.spinner("Fetching fundamental data …"):
        t      = yf.Ticker(ticker)
        info   = t.info
        try:
            cf = t.cashflow          # columns = fiscal years
        except Exception:
            cf = pd.DataFrame()

    shares_outstanding = _safe_float(info.get("sharesOutstanding"), 1e9)
    current_price      = float(df["Close"].squeeze().iloc[-1])

    # ── Extract FCF ──
    # yfinance cashflow rows: "Free Cash Flow" or computed as OCF - Capex
    fcf_hist = []
    if not cf.empty:
        cf_T = cf.T    # rows = years, cols = line items
        for col in cf_T.columns:
            if "Free Cash Flow" in str(col):
                fcf_hist = cf_T[col].dropna().values.tolist()
                break
        if not fcf_hist:
            # Fallback: Operating Cash Flow − Capital Expenditure
            ocf_col  = [c for c in cf_T.columns if "Operating"    in str(c) and "Cash" in str(c)]
            capex_col= [c for c in cf_T.columns if "Capital"      in str(c)]
            if ocf_col and capex_col:
                ocf   = cf_T[ocf_col[0]].dropna().values
                capex = cf_T[capex_col[0]].fillna(0).values
                n     = min(len(ocf), len(capex))
                fcf_hist = (ocf[:n] - np.abs(capex[:n])).tolist()

    # If still empty, use EPS × shares as rough proxy
    if not fcf_hist:
        eps = _safe_float(info.get("trailingEps"), 1.0)
        fcf_hist = [eps * shares_outstanding * (0.8 ** i) for i in range(4, 0, -1)]

    # Use most recent FCF as base; take absolute value (FCF can be negative in growth phase)
    base_fcf = abs(_safe_float(fcf_hist[0] if fcf_hist else 1e9, 1e9))
    if base_fcf < 1:
        base_fcf = 1e9    # floor

    # ── Projection ──
    # Estimate FCF growth = average of last 2 years growth (capped 0–30%)
    if len(fcf_hist) >= 2:
        growths = []
        for i in range(len(fcf_hist) - 1):
            if fcf_hist[i + 1] != 0:
                g = (fcf_hist[i] - fcf_hist[i + 1]) / abs(fcf_hist[i + 1])
                growths.append(g)
        fcf_growth = float(np.clip(np.mean(growths) if growths else 0.08, 0, 0.30))
    else:
        fcf_growth = 0.08    # default 8%

    projected_fcf = []
    pv_fcf        = []
    for yr in range(1, forecast_yrs + 1):
        fcf_yr = base_fcf * (1 + fcf_growth) ** yr
        pv_yr  = fcf_yr   / (1 + wacc) ** yr
        projected_fcf.append(fcf_yr)
        pv_fcf.append(pv_yr)

    # ── Terminal Value (Gordon Growth Model) ──
    terminal_fcf = projected_fcf[-1] * (1 + terminal_g)
    terminal_val = terminal_fcf / (wacc - terminal_g) if wacc > terminal_g else 0
    pv_terminal  = terminal_val / (1 + wacc) ** forecast_yrs

    # ── Enterprise & Intrinsic Value ──
    pv_fcf_total    = sum(pv_fcf)
    enterprise_val  = pv_fcf_total + pv_terminal
    net_debt        = _safe_float(info.get("totalDebt"),     0) \
                    - _safe_float(info.get("totalCash"),     0)
    equity_val      = max(enterprise_val - net_debt, 1)
    intrinsic_value = equity_val / shares_outstanding

    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    verdict = "🟢 Undervalued" if margin_of_safety > 0 else "🔴 Overvalued"

    # ============================================================
    # WATERFALL CHART
    # ============================================================
    years_labels = [f"Y{i}" for i in range(1, forecast_yrs + 1)]
    pv_cr        = [v / 1e7 for v in pv_fcf]   # convert to ₹ Crore for readability

    waterfall_x      = years_labels + ["Terminal PV", "Enterprise Value", "Equity Value"]
    waterfall_measure = (["relative"] * forecast_yrs) + ["relative", "total", "total"]
    waterfall_y      = pv_cr + [pv_terminal / 1e7, None, equity_val / 1e7]

    # plotly waterfall doesn't accept None; replace with actual total
    waterfall_y[-2] = (pv_fcf_total + pv_terminal) / 1e7   # enterprise

    fig = go.Figure(go.Waterfall(
        name="DCF Build-Up",
        orientation="v",
        measure=waterfall_measure,
        x=waterfall_x,
        y=waterfall_y,
        connector=dict(line=dict(color=ACCENT, width=1)),
        increasing=dict(marker_color=GREEN),
        decreasing=dict(marker_color=RED),
        totals=dict(marker_color=YELLOW),
        text=[f"₹{v:.0f}Cr" for v in waterfall_y],
        textposition="outside",
    ))
    fig.update_layout(
        title="DCF Waterfall: Present Value Build-Up (₹ Crore)",
        height=420,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        yaxis_title="₹ Crore",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # PROJECTED FCF TABLE
    # ============================================================
    st.markdown("#### 📋 Projected FCF Schedule")
    rows = []
    for i, (fcf, pv) in enumerate(zip(projected_fcf, pv_fcf), 1):
        rows.append({
            "Year": f"Y{i}",
            "Projected FCF (₹ Cr)": f"{fcf/1e7:,.1f}",
            "PV of FCF (₹ Cr)":    f"{pv/1e7:,.1f}",
        })
    st.table(pd.DataFrame(rows))

    # ============================================================
    # METRICS
    # ============================================================
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏢 Enterprise Value",   f"₹{enterprise_val/1e7:,.0f} Cr")
    c2.metric("💹 Intrinsic Value",    f"₹{intrinsic_value:,.2f}")
    c3.metric("📈 Current Price",      f"₹{current_price:,.2f}")
    c4.metric("🎯 Margin of Safety",   f"{margin_of_safety*100:.1f}%", verdict)

    c5, c6, c7 = st.columns(3)
    c5.metric("📉 FCF Growth Rate",    f"{fcf_growth*100:.1f}%")
    c6.metric("⚙️ WACC",              f"{wacc*100:.1f}%")
    c7.metric("📐 Terminal Growth",    f"{terminal_g*100:.1f}%")

    colour = GREEN if margin_of_safety > 0.20 else (YELLOW if margin_of_safety > 0 else RED)
    st.markdown(
        f"""
        <div style='background:{colour}22; border:1px solid {colour};
                    border-radius:8px; padding:12px 20px; margin-top:12px;'>
            <b style='color:{colour};'>{verdict}</b> — 
            Intrinsic Value <b>₹{intrinsic_value:,.2f}</b> vs 
            Market Price <b>₹{current_price:,.2f}</b> | 
            Margin of Safety <b>{margin_of_safety*100:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
