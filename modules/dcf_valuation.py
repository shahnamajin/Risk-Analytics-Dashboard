# ============================================================
# modules/dcf_valuation.py — Module 4
# Discounted Cash Flow (DCF) Valuation
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time
from .ui_components import (
    chart_container,
    section_divider,
    COLORS,
)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_info_and_cf(tkr: str):
    """Module-level cached fetch — avoids UnserializableReturnValueError."""
    for attempt in range(3):
        try:
            t  = yf.Ticker(tkr)
            try:
                info = dict(t.info)          # convert to plain dict
            except Exception:
                info = {}
            try:
                cf = t.cashflow
            except Exception:
                cf = pd.DataFrame()
            return info, cf
        except Exception:
            time.sleep(2 + attempt * 2)
    return {}, pd.DataFrame()



# Color aliases
ACCENT = COLORS["accent_blue"]
GREEN  = COLORS["accent_green"]
RED    = COLORS["accent_red"]
YELLOW = COLORS["accent_amber"]


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
    """
    Render Module 4 — DCF Valuation.
    
    Discounted Cash Flow analysis using real Free Cash Flow data.
    Estimates intrinsic value by projecting FCF, applying terminal value,
    and discounting at WACC.
    """

    st.markdown("### 💰 DCF Intrinsic Value Estimation")
    st.markdown(
        "Discounted Cash Flow analysis projects Free Cash Flow, applies terminal value "
        "using perpetuity growth model, and discounts at WACC to estimate fair value."
    )
    section_divider()

    with st.spinner("Fetching fundamental data …"):
        info, cf = _fetch_info_and_cf(ticker)

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
        connector=dict(line=dict(color=COLORS["accent_blue"], width=2)),
        increasing=dict(marker_color=COLORS["accent_green"]),
        decreasing=dict(marker_color=COLORS["accent_red"]),
        totals=dict(marker_color=COLORS["accent_amber"]),
        text=[f"₹{v:.0f}Cr" for v in waterfall_y],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>₹%{y:.0f} Crore<extra></extra>",
    ))
    
    fig.update_layout(
        height=450,
        paper_bgcolor=COLORS["primary_dark"],
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            size=11,
            color=COLORS["text_primary"],
        ),
        yaxis_title="₹ Crore (Log Scale)",
        yaxis_type="log",
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
        ),
        xaxis=dict(
            showgrid=False,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=20, b=60),
    )
    
    chart_container("DCF Waterfall: Present Value Build-Up", fig)

    # ============================================================
    # PROJECTED FCF TABLE
    # ============================================================
    st.markdown("### 📋 Projected FCF Schedule")
    st.markdown(f"Forecast period: {forecast_yrs} years | FCF Growth: {fcf_growth*100:.1f}%")
    
    rows = []
    for i, (fcf, pv) in enumerate(zip(projected_fcf, pv_fcf), 1):
        rows.append({
            "Year": f"Year {i}",
            "FCF (₹ Crore)": f"{fcf/1e7:,.1f}",
            "Discount Factor": f"{1/(1 + wacc)**i:.4f}",
            "PV of FCF (₹ Crore)": f"{pv/1e7:,.1f}",
        })
    
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    section_divider()

    # ============================================================
    # VALUATION METRICS
    # ============================================================
    st.markdown("### 📊 DCF Valuation Summary")
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Enterprise Value</div>
                <div class="kpi-value">₹{enterprise_val/1e7:,.0f}Cr</div>
                <div class="kpi-trend" style="color: #8b949e;">PV of Cash Flows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Intrinsic Value</div>
                <div class="kpi-value">₹{intrinsic_value:,.0f}</div>
                <div class="kpi-trend" style="color: #8b949e;">Per Share</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Market Price</div>
                <div class="kpi-value">₹{current_price:,.0f}</div>
                <div class="kpi-trend" style="color: {'#3fb950' if current_price < intrinsic_value else '#f85149'};">
                    Current
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    mos_color = "green" if margin_of_safety > 0.20 else ("amber" if margin_of_safety > 0 else "red")
    with cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Margin of Safety</div>
                <div class="kpi-value">{margin_of_safety*100:.1f}%</div>
                <div class="kpi-trend" style="color: {'#3fb950' if margin_of_safety > 0.20 else '#d29922' if margin_of_safety > 0 else '#f85149'};">
                    {'Undervalued' if margin_of_safety > 0 else 'Overvalued'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_divider()
    
    # Additional metrics
    cols = st.columns(3)
    cols[0].metric("FCF Growth Rate", f"{fcf_growth*100:.1f}%", help="Historical avg growth")
    cols[1].metric("WACC (Discount Rate)", f"{wacc*100:.1f}%", help="Cost of capital")
    cols[2].metric("Terminal Growth", f"{terminal_g*100:.1f}%", help="Perpetual growth rate")

    section_divider()

    # Verdict box
    colour = COLORS["accent_green"] if margin_of_safety > 0.20 else (COLORS["accent_amber"] if margin_of_safety > 0 else COLORS["accent_red"])
    verdict_text = (
        "🟢 Strongly Undervalued" if margin_of_safety > 0.20
        else ("🟡 Moderately Undervalued" if margin_of_safety > 0.05
        else ("🟠 Slightly Undervalued" if margin_of_safety > 0
        else "🔴 Overvalued"))
    )
    
    st.markdown(
        f"""
        <div style='background:{colour}15; border:2px solid {colour};
                    border-radius:10px; padding:1rem; margin-top:1rem;'>
            <div style='color:{colour}; font-size:1.1rem; font-weight:700; margin-bottom:0.5rem;'>
                {verdict_text}
            </div>
            <div style='color:#8b949e; font-size:0.9rem; line-height:1.6;'>
                <b>Intrinsic Value:</b> ₹{intrinsic_value:,.2f} 
                &nbsp; | &nbsp;
                <b>Market Price:</b> ₹{current_price:,.2f}<br>
                <b>Margin of Safety:</b> {margin_of_safety*100:+.1f}%
                &nbsp; | &nbsp;
                <b>Spread:</b> ₹{abs(intrinsic_value - current_price):,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )