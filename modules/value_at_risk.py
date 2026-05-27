# ============================================================
# modules/value_at_risk.py — Module 6
# Value at Risk (VaR) & Conditional VaR (CVaR)
# ============================================================
# Three VaR methods:
#   Historical   — empirical quantile of past returns
#   Parametric   — assumes Normal distribution
#   Monte Carlo  — GBM simulation quantile
#
# CVaR (Expected Shortfall) is the AVERAGE loss beyond VaR.
#
# Kupiec Test validates whether observed exceptions match
# the expected exception rate.
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"


def _historical_var(returns: np.ndarray, alpha: float) -> float:
    """Historical simulation VaR at confidence level alpha."""
    return float(np.percentile(returns, (1 - alpha) * 100))


def _parametric_var(returns: np.ndarray, alpha: float) -> float:
    """Parametric (Normal) VaR."""
    mu, sigma = returns.mean(), returns.std()
    return float(mu + sigma * stats.norm.ppf(1 - alpha))


def _mc_var(mu_d: float, sigma_d: float, alpha: float,
            n_sims: int = 10000, seed: int = 42) -> float:
    """Monte Carlo VaR via GBM daily log returns."""
    rng       = np.random.default_rng(seed)
    sim_rets  = mu_d + sigma_d * rng.standard_normal(n_sims)
    return float(np.percentile(sim_rets, (1 - alpha) * 100))


def _cvar(returns: np.ndarray, var: float) -> float:
    """CVaR = mean of returns that fall below VaR."""
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def _kupiec_test(returns: np.ndarray, var: float, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (PoF) Test.
    H0: Observed exception rate == (1 - alpha).
    Returns exceptions, expected, p-value, and pass/fail.
    """
    n          = len(returns)
    exceptions = int(np.sum(returns < var))
    expected   = n * (1 - alpha)
    p_hat      = exceptions / n if n > 0 else 0
    p0         = 1 - alpha

    # LR statistic (log-likelihood ratio)
    if p_hat > 0 and p_hat < 1:
        lr = -2 * (
            np.log((1 - p0) ** (n - exceptions) * p0 ** exceptions) -
            np.log((1 - p_hat) ** (n - exceptions) * p_hat ** exceptions)
        )
    else:
        lr = 0.0

    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "Exceptions":       exceptions,
        "Expected":         f"{expected:.1f}",
        "LR Statistic":     f"{lr:.3f}",
        "p-value":          f"{p_value:.4f}",
        "Result":           "✅ Valid" if p_value > 0.05 else "❌ Invalid",
    }


def render_value_at_risk(df, ticker, **kwargs):
    """Render Module 6 — Value at Risk."""

    st.subheader("⚠️ Value at Risk (VaR) Analysis")
    st.markdown(
        "VaR estimates the **maximum expected loss** over 1 day at a given confidence level. "
        "Three methods are compared; Kupiec test validates each VaR model."
    )

    close       = df["Close"].squeeze()
    log_returns = np.log(close / close.shift(1)).dropna().values
    mu_d        = float(log_returns.mean())
    sigma_d     = float(log_returns.std())

    # ── Compute VaR for 95% and 99% ──
    results = {}
    for alpha in [0.95, 0.99]:
        h_var  = _historical_var(log_returns, alpha)
        p_var  = _parametric_var(log_returns, alpha)
        mc_var = _mc_var(mu_d, sigma_d, alpha)

        h_cvar  = _cvar(log_returns, h_var)
        p_cvar  = _cvar(log_returns, p_var)
        mc_cvar = _cvar(log_returns, mc_var)

        results[alpha] = {
            "Historical":  (h_var,  h_cvar),
            "Parametric":  (p_var,  p_cvar),
            "Monte Carlo": (mc_var, mc_cvar),
        }

    # ── Comparison Table ──
    st.markdown("#### 📊 VaR Comparison Table")
    rows = []
    for method in ["Historical", "Parametric", "Monte Carlo"]:
        row = {"Method": method}
        for alpha, label in [(0.95, "95%"), (0.99, "99%")]:
            v, c = results[alpha][method]
            row[f"VaR {label}"]  = f"{v*100:.3f}%"
            row[f"CVaR {label}"] = f"{c*100:.3f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)

    # ── Kupiec Test ──
    st.markdown("#### 🔬 Kupiec Backtesting (Historical VaR)")
    kup_rows = []
    for alpha, label in [(0.95, "95%"), (0.99, "99%")]:
        h_var = results[alpha]["Historical"][0]
        k     = _kupiec_test(log_returns, h_var, alpha)
        k["Confidence"] = label
        kup_rows.append(k)
    st.dataframe(pd.DataFrame(kup_rows).set_index("Confidence"), use_container_width=True)

    # ── Loss Distribution Histogram ──
    h_var95 = results[0.95]["Historical"][0]
    h_var99 = results[0.99]["Historical"][0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_returns * 100,
        nbinsx=80,
        marker_color=ACCENT,
        opacity=0.7,
        name="Daily Log Returns",
    ))
    fig.add_vline(x=h_var95 * 100, line_dash="dash", line_color=YELLOW,
                  annotation_text="VaR 95%", annotation_font_color=YELLOW)
    fig.add_vline(x=h_var99 * 100, line_dash="dash", line_color=RED,
                  annotation_text="VaR 99%", annotation_font_color=RED)

    # Shade the tail
    tail_x = [r * 100 for r in log_returns if r <= h_var95]
    if tail_x:
        fig.add_trace(go.Histogram(
            x=tail_x, nbinsx=30,
            marker_color=RED, opacity=0.5,
            name="Tail (< VaR 95%)",
        ))

    fig.update_layout(
        title="Daily Log Return Distribution with VaR Thresholds",
        xaxis_title="Log Return (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=380,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI Metrics ──
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚠️ Historical VaR 95%",  f"{h_var95*100:.3f}%")
    c2.metric("🚨 Historical VaR 99%",  f"{h_var99*100:.3f}%")
    c3.metric("📉 CVaR 95%",
              f"{results[0.95]['Historical'][1]*100:.3f}%")
    c4.metric("💀 CVaR 99%",
              f"{results[0.99]['Historical'][1]*100:.3f}%")

    st.info(
        "ℹ️  **Interpretation:** A VaR of **-X%** at 95% confidence means on "
        "95% of days, losses will NOT exceed X%. CVaR is the average loss on the "
        "worst 5% of days. A ✅ Kupiec result means the VaR model is statistically valid."
    )
