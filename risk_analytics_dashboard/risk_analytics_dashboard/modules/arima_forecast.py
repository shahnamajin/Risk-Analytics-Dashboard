# ============================================================
# modules/arima_forecast.py — Module 2
# ARIMA Price Forecasting with Walk-Forward Validation
# ============================================================
# Uses pmdarima's auto_arima to automatically find the best
# ARIMA(p,d,q) order, then forecasts 90 trading days ahead.
# Walk-forward validation splits data 80/20 and measures
# accuracy with RMSE, MAE, and MAPE.
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


# ── Helper: compute error metrics ──
def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = actual - predicted
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    mae    = float(np.mean(np.abs(errors)))
    mape   = float(np.mean(np.abs(errors / actual)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


@st.cache_data(show_spinner=False, ttl=3600)
def _fit_arima(prices_list: list):
    """
    Fit auto_arima on log-transformed prices.
    We cache this because fitting ARIMA is expensive (~10–30 s).
    Streamlit cannot cache DataFrame objects directly in all versions,
    so we pass a plain list and convert inside.
    """
    from pmdarima import auto_arima

    prices = np.array(prices_list)
    log_p  = np.log(prices)

    model = auto_arima(
        log_p,
        stepwise=True,
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
    )
    return model


def render_arima_forecast(df, ticker, **kwargs):
    """Render Module 2 — ARIMA Forecasting."""

    st.subheader("📈 ARIMA Price Forecasting")
    st.markdown(
        "Auto-selected ARIMA model forecasts **90 trading days** ahead "
        "with 95 % confidence intervals and walk-forward validation."
    )

    close = df["Close"].squeeze()

    # ── Fit model ──
    with st.spinner("Fitting ARIMA model (this may take ~15 s) …"):
        model = _fit_arima(close.values.tolist())

    order = model.order          # e.g. (2, 1, 2)
    aic   = model.aic()
    bic   = model.bic()

    # ── Forecast 90 days ──
    HORIZON = 90
    log_forecast, conf_int = model.predict(
        n_periods=HORIZON, return_conf_int=True, alpha=0.05
    )
    forecast_prices = np.exp(log_forecast)
    lower           = np.exp(conf_int[:, 0])
    upper           = np.exp(conf_int[:, 1])

    # ── Build future date index (skip weekends naïvely) ──
    last_date  = close.index[-1]
    fut_dates  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)

    # ── Walk-Forward Validation (80/20 split) ──
    split_idx   = int(len(close) * 0.80)
    train_close = close.iloc[:split_idx]
    test_close  = close.iloc[split_idx:]

    with st.spinner("Running walk-forward validation …"):
        from pmdarima import auto_arima as _aa
        wf_model = _aa(
            np.log(train_close.values),
            stepwise=True, seasonal=False,
            error_action="ignore", suppress_warnings=True,
        )
        wf_preds_log, _ = wf_model.predict(
            n_periods=len(test_close), return_conf_int=True
        )
        wf_preds = np.exp(wf_preds_log)

    m = _metrics(test_close.values, wf_preds)

    # ── Direction ──
    direction = "📈 Uptrend" if forecast_prices[-1] > float(close.iloc[-1]) else "📉 Downtrend"

    # ============================================================
    # CHARTS
    # ============================================================
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Historical Price + ARIMA Forecast", "Walk-Forward Validation"],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.10,
    )

    # Historical
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, name="Historical",
        line=dict(color=ACCENT, width=1.5),
    ), row=1, col=1)

    # Forecast
    fig.add_trace(go.Scatter(
        x=fut_dates, y=forecast_prices, name="Forecast",
        line=dict(color=GREEN, width=2, dash="dot"),
    ), row=1, col=1)

    # Confidence interval ribbon
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(upper)     + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(63,185,80,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ), row=1, col=1)

    # Walk-forward actual
    fig.add_trace(go.Scatter(
        x=test_close.index, y=test_close.values, name="Actual (test)",
        line=dict(color=ACCENT, width=1.5),
    ), row=2, col=1)

    # Walk-forward predicted
    fig.add_trace(go.Scatter(
        x=test_close.index, y=wf_preds, name="Predicted (test)",
        line=dict(color=YELLOW, width=1.5, dash="dash"),
    ), row=2, col=1)

    fig.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # METRICS TABLE
    # ============================================================
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔢 Model Order",    f"ARIMA{order}")
    col2.metric("📊 AIC",            f"{aic:.1f}")
    col3.metric("📊 BIC",            f"{bic:.1f}")
    col4.metric("🎯 RMSE",           f"₹{m['RMSE']:.2f}")
    col5.metric("📐 MAE",            f"₹{m['MAE']:.2f}")

    col6, col7, col8 = st.columns(3)
    col6.metric("📉 MAPE",           f"{m['MAPE (%)']:.2f}%")
    col7.metric("🔮 Forecasted (90d)", f"₹{forecast_prices[-1]:,.2f}")
    col8.metric("📈 Direction",      direction)

    st.info(
        f"ℹ️  **Interpretation:** The model predicts the price will reach "
        f"**₹{forecast_prices[-1]:,.2f}** in ~90 trading days ({direction}). "
        f"MAPE of **{m['MAPE (%)']:.1f}%** indicates model accuracy on unseen data."
    )
