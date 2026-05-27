# ============================================================
# modules/garch_volatility.py — Module 3
# GARCH(1,1) Volatility Modelling
# ============================================================
# GARCH (Generalised AutoRegressive Conditional
# Heteroskedasticity) captures volatility clustering —
# the tendency for high-volatility periods to persist.
#
# GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"


def render_garch_volatility(df, ticker, **kwargs):
    """Render Module 3 — GARCH Volatility."""

    st.subheader("🌪️ GARCH(1,1) Volatility Modelling")
    st.markdown(
        "GARCH(1,1) estimates **conditional volatility** — how uncertainty "
        "changes over time. High-volatility regimes are flagged automatically."
    )

    close       = df["Close"].squeeze()
    log_returns = (np.log(close / close.shift(1)).dropna() * 100)   # in %

    # ── Fit GARCH(1,1) ──
    with st.spinner("Fitting GARCH(1,1) model …"):
        from arch import arch_model
        am    = arch_model(log_returns, vol="Garch", p=1, q=1,
                           dist="normal", rescale=False)
        res   = am.fit(disp="off")

    cond_vol      = res.conditional_volatility           # daily % σ
    annual_vol    = cond_vol * np.sqrt(252)              # annualised %
    current_vol   = float(cond_vol.iloc[-1]) * np.sqrt(252)
    long_term_avg = float(cond_vol.mean())   * np.sqrt(252)

    # ── Regime classification ──
    p75  = float(annual_vol.quantile(0.75))
    p25  = float(annual_vol.quantile(0.25))
    last = float(annual_vol.iloc[-1])

    if last > p75:
        regime, regime_color = "🔴 High Volatility",     RED
    elif last < p25:
        regime, regime_color = "🟢 Low Volatility",      GREEN
    else:
        regime, regime_color = "🟡 Moderate Volatility", YELLOW

    # ── Find latest volatility spike (above 75th pct) ──
    spikes = annual_vol[annual_vol > p75]
    # Find the SINGLE highest spike date (3b requirement)
    if not spikes.empty:
        spike_date_ts = annual_vol.idxmax()
        spike_date = str(spike_date_ts.date())
    else:
        spike_date_ts = None
        spike_date = "N/A"

    # ── Rolling 20-day historical volatility ──
    hist_vol_20 = log_returns.rolling(20).std() * np.sqrt(252)

    # ============================================================
    # CHARTS
    # ============================================================
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Log Returns (%)",
            "GARCH Conditional Volatility (Annualised %)",
            "Rolling 20-Day Historical Volatility (Annualised %)",
        ],
        row_heights=[0.3, 0.4, 0.3],
        vertical_spacing=0.09,
    )

    # Returns bar chart
    colors = [GREEN if v >= 0 else RED for v in log_returns.values]
    fig.add_trace(go.Bar(
        x=log_returns.index, y=log_returns.values,
        marker_color=colors, name="Log Returns",
        showlegend=False,
    ), row=1, col=1)

    # GARCH conditional volatility
    fig.add_trace(go.Scatter(
        x=annual_vol.index, y=annual_vol.values,
        line=dict(color=YELLOW, width=1.5), name="GARCH Vol",
        fill="tozeroy", fillcolor="rgba(210,153,34,0.08)",
    ), row=2, col=1)

    # 75th percentile threshold line
    fig.add_hline(
        y=p75, line_dash="dash", line_color=RED,
        annotation_text="75th pct", row=2, col=1,
    )
    fig.add_hline(
        y=p25, line_dash="dash", line_color=GREEN,
        annotation_text="25th pct", row=2, col=1,
    )
    # Mark highest volatility spike with vertical dashed line (3b)
    # add_vline does not support row/col — use add_shape for subplot targeting
    if spike_date_ts is not None:
        try:
            x_val = spike_date_ts.timestamp() * 1000   # plotly uses ms timestamps
            fig.add_shape(
                type="line",
                x0=x_val, x1=x_val,
                y0=0, y1=1,
                xref="x2", yref="y2 domain",
                line=dict(color=RED, width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=spike_date_ts, y=1,
                xref="x2", yref="y2 domain",
                text=f"Peak: {spike_date}",
                showarrow=False,
                font=dict(color=RED, size=10),
                xanchor="left", yanchor="bottom",
            )
        except Exception:
            pass   # skip annotation if timestamp conversion fails

    # Rolling historical vol
    fig.add_trace(go.Scatter(
        x=hist_vol_20.index, y=hist_vol_20.values,
        line=dict(color=ACCENT, width=1.5), name="20-Day HVol",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ), row=3, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # METRICS
    # ============================================================
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Current Vol (Ann.)",  f"{current_vol*100:.1f}%")
    c2.metric("📉 Long-Term Avg Vol",   f"{long_term_avg*100:.1f}%")
    c3.metric("🌡️ Regime",             regime)
    c4.metric("🗓️ Latest Spike Date",   spike_date)

    # GARCH parameters
    st.markdown("#### GARCH(1,1) Parameters")
    params = res.params
    pcols  = st.columns(4)
    param_names = ["omega (ω)", "alpha[1] (α)", "beta[1] (β)", "Persistence (α+β)"]
    param_vals  = [
        params.get("omega", 0),
        params.get("alpha[1]", 0),
        params.get("beta[1]", 0),
        params.get("alpha[1]", 0) + params.get("beta[1]", 0),
    ]
    for col, name, val in zip(pcols, param_names, param_vals):
        col.metric(name, f"{val:.4f}")

    st.info(
        "ℹ️  **Interpretation:** Persistence (α+β) close to 1 means shocks to "
        "volatility die out slowly — volatility clustering is strong. "
        f"Current regime: **{regime}** (annualised GARCH vol = {current_vol*100:.1f}%)."
    )