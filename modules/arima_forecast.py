# ============================================================
# modules/arima_forecast.py — Module 2
# ARIMA Price Forecasting with Walk-Forward Validation
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .ui_components import (
    chart_container,
    section_divider,
    COLORS,
)

# Color aliases for convenience
ACCENT = COLORS["accent_blue"]
GREEN  = COLORS["accent_green"]
RED    = COLORS["accent_red"]
YELLOW = COLORS["accent_amber"]


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
    """
    Render Module 2 — ARIMA Forecasting.
    
    Auto-selected ARIMA model forecasts 90 trading days ahead
    with 95% confidence intervals and walk-forward validation.
    """

    st.markdown("### 📈 ARIMA Price Forecasting")
    st.markdown(
        "Auto-selected ARIMA model forecasts **90 trading days** ahead "
        "with 95% confidence intervals and walk-forward validation metrics."
    )
    
    section_divider()

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
        line=dict(color=COLORS["accent_blue"], width=1.5),
        hovertemplate="<b>Historical</b><br>%{x|%b %d}<br>₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Forecast
    fig.add_trace(go.Scatter(
        x=fut_dates, y=forecast_prices, name="Forecast",
        line=dict(color=COLORS["accent_green"], width=2, dash="dash"),
        hovertemplate="<b>Forecast</b><br>%{x|%b %d}<br>₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Confidence interval ribbon
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(upper)     + list(lower[::-1]),
        fill="toself",
        fillcolor=f"rgba(63,185,80,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
        hovertemplate="<b>CI Band</b><br>%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Walk-forward actual
    fig.add_trace(go.Scatter(
        x=test_close.index, y=test_close.values, name="Actual (Test)",
        line=dict(color=COLORS["accent_blue"], width=1.5),
        hovertemplate="<b>Actual</b><br>%{x|%b %d}<br>₹%{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    # Walk-forward predicted
    fig.add_trace(go.Scatter(
        x=test_close.index, y=wf_preds, name="Predicted (Test)",
        line=dict(color=COLORS["accent_amber"], width=1.5, dash="dash"),
        hovertemplate="<b>Predicted</b><br>%{x|%b %d}<br>₹%{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    # Update layout with professional styling
    fig.update_layout(
        height=650,
        paper_bgcolor=COLORS["primary_dark"],
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            size=11,
            color=COLORS["text_primary"],
        ),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(13, 17, 23, 0.8)",
            bordercolor=COLORS["border_subtle"],
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
        ),
        xaxis2=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
        ),
        yaxis2=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(48, 54, 61, 0.2)",
        ),
    )
    
    chart_container("ARIMA Forecast with Walk-Forward Validation", fig)

    # ============================================================
    # METRICS & INTERPRETATION
    # ============================================================
    
    section_divider()
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model Order", f"ARIMA{order}")
    col2.metric("AIC Score", f"{aic:.1f}")
    col3.metric("BIC Score", f"{bic:.1f}")
    col4.metric("RMSE (₹)", f"{m['RMSE']:.2f}")
    col5.metric("MAE (₹)", f"{m['MAE']:.2f}")

    col6, col7, col8 = st.columns(3)
    col6.metric("MAPE (%)", f"{m['MAPE (%)']:.2f}%")
    col7.metric("90-Day Target", f"₹{forecast_prices[-1]:,.0f}")
    col8.metric("Direction", direction)

    section_divider()
    
    # Interpretation box
    st.markdown("### 🔍 Model Interpretation")
    
    accuracy_level = (
        "🟢 Excellent (< 5%)" if m['MAPE (%)'] < 5
        else "🟡 Good (5-10%)" if m['MAPE (%)'] < 10
        else "🟠 Fair (10-20%)" if m['MAPE (%)'] < 20
        else "🔴 Poor (> 20%)"
    )
    
    st.info(
        f"""
        **ARIMA{order}** Model Analysis:
        
        • **Forecast Target:** ₹{forecast_prices[-1]:,.0f} in 90 trading days ({direction})
        • **Model Accuracy (MAPE):** {m['MAPE (%)']:.2f}% — {accuracy_level}
        • **Error Metrics:** RMSE ₹{m['RMSE']:.2f} | MAE ₹{m['MAE']:.2f}
        • **Confidence Interval:** 95% band shown in chart above
        • **Validation Method:** Walk-forward (80% train, 20% test)
        
        The model captures temporal autocorrelation in price movements.
        MAPE indicates expected accuracy on future unseen data.
        """
    )
