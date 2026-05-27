# ============================================================
# modules/correlation_heatmap.py — Module 10
# Correlation Heatmap of Log Returns
# ============================================================
# Daily log returns are computed for all 5 tickers.
# A Pearson correlation matrix is plotted as a heatmap.
# Pairs with |corr| > 0.70 are flagged as highly correlated
# (risk of undiversification).
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"

TICKER_NAMES = {
    "INFY.NS":     "Infosys",
    "RELIANCE.NS": "Reliance",
    "ITC.NS":      "ITC",
    "HDFCBANK.NS": "HDFC Bank",
    "ABB.NS":      "ABB India",
}


def render_correlation_heatmap(all_close, **kwargs):
    """Render Module 10 — Correlation Heatmap."""

    st.subheader("🔥 Return Correlation Heatmap")
    st.markdown(
        "Pearson correlation of **daily log returns** across all 5 stocks. "
        "High positive correlation (>0.70) means stocks move together — "
        "**less diversification benefit**."
    )

    # ── Compute log returns ──
    log_rets = np.log(all_close / all_close.shift(1)).dropna()

    # Rename columns for readability
    rename = {t: TICKER_NAMES.get(t, t) for t in log_rets.columns}
    log_rets.rename(columns=rename, inplace=True)

    corr = log_rets.corr()
    tickers_clean = list(corr.columns)
    corr_matrix   = corr.values

    # ── Heatmap ──
    # Diverging: red = +1, white = 0, blue = -1
    z_text = [[f"{corr_matrix[i][j]:.2f}" for j in range(len(tickers_clean))]
               for i in range(len(tickers_clean))]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=tickers_clean,
        y=tickers_clean,
        colorscale=[
            [0.0, "#1168e0"],   # deep blue  = -1
            [0.5, "#ffffff"],   # white      =  0
            [1.0, "#d62728"],   # deep red   = +1
        ],
        zmin=-1, zmax=1,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#0d1117"},
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    fig.update_layout(
        title="Pearson Correlation Matrix — Daily Log Returns",
        height=500,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=100, r=80, t=60, b=80),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── High Correlation Warnings ──
    st.markdown("---")
    st.markdown("#### ⚠️ High Correlation Pairs (|r| > 0.70)")
    high_corr = []
    n = len(tickers_clean)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i][j]
            if abs(r) > 0.70:
                high_corr.append({
                    "Pair":        f"{tickers_clean[i]} × {tickers_clean[j]}",
                    "Correlation": f"{r:.3f}",
                    "Warning":     "🔴 Highly Correlated" if r > 0 else "🔵 Strong Negative",
                })

    if high_corr:
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    else:
        st.success("✅ No highly correlated pairs found (|r| ≤ 0.70). Portfolio is well-diversified.")

    # ── Most diversifying & most redundant pair ──
    # Off-diagonal pairs only
    off_diag = [(i, j, corr_matrix[i][j])
                for i in range(n) for j in range(i + 1, n)]

    if off_diag:
        min_pair = min(off_diag, key=lambda x: x[2])
        max_pair = max(off_diag, key=lambda x: x[2])

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style='background:#3fb95022; border:1px solid #3fb950;
                            border-radius:8px; padding:14px;'>
                    <b style='color:#3fb950;'>🌿 Most Diversifying Pair</b><br>
                    <span style='font-size:20px;'>
                        {tickers_clean[min_pair[0]]} × {tickers_clean[min_pair[1]]}
                    </span><br>
                    Correlation: <b>{min_pair[2]:.3f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style='background:#f8514922; border:1px solid #f85149;
                            border-radius:8px; padding:14px;'>
                    <b style='color:#f85149;'>🔗 Most Redundant Pair</b><br>
                    <span style='font-size:20px;'>
                        {tickers_clean[max_pair[0]]} × {tickers_clean[max_pair[1]]}
                    </span><br>
                    Correlation: <b>{max_pair[2]:.3f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Rolling correlation (first two tickers) ──
    st.markdown("---")
    st.markdown("#### 📈 Rolling 60-Day Correlation")
    if len(log_rets.columns) >= 2:
        t1, t2 = log_rets.columns[0], log_rets.columns[1]
        roll_corr = log_rets[t1].rolling(60).corr(log_rets[t2]).dropna()

        rcfig = go.Figure(go.Scatter(
            x=roll_corr.index,
            y=roll_corr.values,
            mode="lines",
            line=dict(color=ACCENT, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            name=f"{t1} × {t2}",
        ))
        rcfig.add_hline(y=0.70,  line_dash="dash", line_color=RED,
                        annotation_text="0.70 threshold", annotation_font_color=RED)
        rcfig.add_hline(y=-0.70, line_dash="dash", line_color=YELLOW,
                        annotation_text="-0.70 threshold", annotation_font_color=YELLOW)
        rcfig.update_layout(
            title=f"Rolling 60-Day Correlation: {t1} × {t2}",
            xaxis_title="Date",
            yaxis_title="Pearson r",
            yaxis=dict(range=[-1, 1]),
            height=300,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            margin=dict(l=60, r=20, t=50, b=40),
        )
        st.plotly_chart(rcfig, use_container_width=True)
