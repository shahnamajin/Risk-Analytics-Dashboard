# ============================================================
# modules/credit_risk.py — Module 7
# Credit Risk Model — Probability of Default
# ============================================================
# A Logistic Regression model is trained on 500 synthetic
# company records (with realistic financial ratios).
# The selected stock's fundamentals are used to predict its
# Probability of Default (PD).
#
# Features: D/E, Interest Coverage, Current Ratio, ROE, NPM
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics       import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import time
from .ui_components import render_html

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"


def _generate_synthetic_data(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """
    Create a synthetic dataset of n companies with 5 financial features.
    Default label (1) is assigned based on a logistic rule so the data
    is balanced and realistic.
    """
    rng = np.random.default_rng(seed)

    de   = rng.exponential(scale=1.5,  size=n).clip(0, 10)    # Debt/Equity
    ic   = rng.normal(loc=5,  scale=3, size=n).clip(0, 20)    # Interest Coverage
    cr   = rng.normal(loc=2,  scale=1, size=n).clip(0.5, 6)   # Current Ratio
    roe  = rng.normal(loc=12, scale=8, size=n).clip(-30, 40)  # ROE %
    npm  = rng.normal(loc=8,  scale=5, size=n).clip(-20, 30)  # Net Profit Margin %

    # Probability of default increases with D/E, decreases with IC/CR/ROE/NPM
    log_odds = -2 + 0.4 * de - 0.3 * ic - 0.5 * cr - 0.05 * roe - 0.08 * npm
    prob_def = 1 / (1 + np.exp(-log_odds))
    default  = (rng.random(n) < prob_def).astype(int)

    return pd.DataFrame({
        "Debt_to_Equity":      de,
        "Interest_Coverage":   ic,
        "Current_Ratio":       cr,
        "ROE":                 roe,
        "Net_Profit_Margin":   npm,
        "Default":             default,
    })


def _credit_grade(pd_pct: float) -> tuple:
    """Map PD % to credit grade and colour."""
    thresholds = [(1, "AAA", "#3fb950"), (3, "AA", "#58a6ff"),
                  (5, "A",  "#bc8cff"), (8, "BBB", "#d29922"),
                  (12,"BB", "#f0883e"), (20,"B",   "#f85149")]
    for limit, grade, colour in thresholds:
        if pd_pct <= limit:
            return grade, colour
    return "D", "#ff3f3f"


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except Exception:
        return default



@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cr_info(tkr: str) -> dict:
    """Module-level cached fundamentals fetch."""
    for attempt in range(3):
        try:
            return dict(yf.Ticker(tkr).info)
        except Exception:
            time.sleep(2 + attempt)
    return {}

def render_credit_risk(df, ticker, **kwargs):
    """Render Module 7 — Credit Risk."""

    st.subheader("🏦 Credit Risk — Probability of Default")
    st.markdown(
        "A Logistic Regression model trained on **500 synthetic company records** "
        "predicts the Probability of Default (PD) for the selected stock."
    )

    # ── Synthetic dataset ──
    data    = _generate_synthetic_data(500)
    X       = data.drop("Default", axis=1).values
    y       = data["Default"].values
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_tr, y_tr)

    with st.spinner("Fetching fundamentals …"):
        info = _fetch_cr_info(ticker)

    stock_de  = _safe_float(info.get("debtToEquity",           1.5))
    stock_ic  = _safe_float(info.get("ebitda") and
                             info.get("totalDebt") and
                             info["ebitda"] / max(info["totalDebt"], 1), 5)
    stock_cr  = _safe_float(info.get("currentRatio",           1.5))
    stock_roe = _safe_float(info.get("returnOnEquity",         0.10)) * 100
    stock_npm = _safe_float(info.get("profitMargins",          0.08)) * 100

    stock_feat = np.array([[stock_de, stock_ic, stock_cr, stock_roe, stock_npm]])
    stock_feat_sc = scaler.transform(stock_feat)
    pd_prob    = float(model.predict_proba(stock_feat_sc)[0][1])
    pd_pct     = pd_prob * 100

    grade, grade_colour = _credit_grade(pd_pct)
    credit_score = max(300, min(900, int(900 - pd_pct * 30)))   # proxy CIBIL-style

    # ── Semicircle Gauge ──
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pd_pct,
        number={"suffix": "%", "font": {"size": 32, "color": "#e6edf3"}},
        title={"text": f"Probability of Default — Grade {grade}",
               "font": {"color": grade_colour}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#e6edf3"},
            "bar":   {"color": grade_colour},
            "bgcolor": "#1c2128",
            "steps": [
                {"range": [0,  5],  "color": "rgba(63,185,80,0.25)"},
                {"range": [5,  15], "color": "rgba(210,153,34,0.25)"},
                {"range": [15, 100],"color": "rgba(248,81,73,0.25)"},
            ],
            "threshold": {
                "line":  {"color": RED, "width": 3},
                "thickness": 0.75,
                "value": 15,
            },
        },
    ))
    gauge_fig.update_layout(
        height=280,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        margin=dict(l=30, r=30, t=60, b=10),
    )

    # ── Confusion Matrix ──
    y_pred = model.predict(X_te)
    cm     = confusion_matrix(y_te, y_pred)
    cm_fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: No Default", "Pred: Default"],
        y=["Actual: No Default", "Actual: Default"],
        colorscale="Blues",
        text=cm.astype(str),
        texttemplate="%{text}",
        showscale=False,
    ))
    cm_fig.update_layout(
        title="Confusion Matrix",
        height=280,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        margin=dict(l=50, r=20, t=50, b=50),
    )

    # ── ROC Curve ──
    y_prob    = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc   = auc(fpr, tpr)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=ACCENT, width=2),
        name=f"ROC (AUC={roc_auc:.3f})",
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=YELLOW, dash="dash"), name="Random",
    ))
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=280,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=50, r=20, t=50, b=50),
    )

    # ── PD Trend (simulated 15 months) ──
    np.random.seed(42)
    pd_trend = pd_pct + np.cumsum(np.random.normal(0, 0.3, 15))
    pd_trend = np.clip(pd_trend, 0.1, 99)
    months   = [f"M-{14-i}" for i in range(15)]

    trend_fig = go.Figure(go.Scatter(
        x=months, y=pd_trend, mode="lines+markers",
        line=dict(color=PURPLE, width=2),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(188,140,255,0.10)",
        name="PD Trend",
    ))
    trend_fig.update_layout(
        title="PD Trend — Last 15 Months (Simulated)",
        xaxis_title="Month",
        yaxis_title="PD (%)",
        height=280,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=50, r=20, t=50, b=50),
    )

    # ── Layout ──
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(gauge_fig, use_container_width=True)
    with col2: st.plotly_chart(roc_fig,   use_container_width=True)

    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(cm_fig,    use_container_width=True)
    with col4: st.plotly_chart(trend_fig, use_container_width=True)

    # ── Metrics ──
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚠️ Prob. of Default",  f"{pd_pct:.2f}%")
    c2.metric("🏦 Credit Score",      f"{credit_score}")
    c3.metric("🏅 Credit Grade",      grade)
    c4.metric("📊 AUC Score",         f"{roc_auc:.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("📈 Debt/Equity",        f"{stock_de:.2f}")
    c6.metric("🛡️ Interest Coverage", f"{stock_ic:.2f}x")
    c7.metric("💧 Current Ratio",      f"{stock_cr:.2f}")

    colour = GREEN if pd_pct < 5 else (YELLOW if pd_pct < 15 else RED)
    render_html(
        f"""<div style='background:{colour}22; border:1px solid {colour};
                        border-radius:8px; padding:12px 20px; margin-top:12px;'>
                <b style='color:{colour};'>Grade {grade}</b> — 
                PD = <b>{pd_pct:.2f}%</b> | Credit Score = <b>{credit_score}</b>
            </div>""",
    )
    