# ============================================================
# app.py — Main Entry Point
# Risk Analytics Dashboard | MCA Capstone Project
# Bloomberg/Trading Terminal Style
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import os

# ── Page config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide Streamlit toolbar and footer ──
st.markdown("""<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# ── Load custom CSS ──
css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_file_path):
    with open(css_file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Import UI components ──
from modules.ui_components import (
    render_header,
    render_footer,
    sidebar_section_header,
    sidebar_section_divider,
    section_divider,
)

# ── Import all dashboard modules ──
from modules.executive_summary    import render_executive_summary
from modules.arima_forecast       import render_arima_forecast
from modules.garch_volatility     import render_garch_volatility
from modules.dcf_valuation        import render_dcf_valuation
from modules.monte_carlo          import render_monte_carlo
from modules.value_at_risk        import render_value_at_risk
from modules.credit_risk          import render_credit_risk
from modules.portfolio_optimization import render_portfolio_optimization
from modules.stress_testing       import render_stress_testing
from modules.correlation_heatmap  import render_correlation_heatmap


# ============================================================
# SIDEBAR — Professional Dashboard Controls
# ============================================================

# Logo
st.sidebar.markdown(
    '<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #30363d; margin-bottom: 1.5rem;">'
    '<span style="font-size: 1.5rem; font-weight: 700; color: #58a6ff;">📊</span>'
    '<p style="margin: 0.5rem 0 0; color: #8b949e; font-size: 0.85rem; font-weight: 600;">Risk Analytics</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Dashboard Controls Section ──
sidebar_section_header("Dashboard Controls", "🎯")

TICKERS = ["INFY.NS", "RELIANCE.NS", "ITC.NS", "HDFCBANK.NS", "ABB.NS"]
selected_ticker = st.sidebar.selectbox(
    "Stock Selection",
    TICKERS,
    index=0,
    help="Choose the NSE stock to analyse across all modules.",
    label_visibility="collapsed",
)

default_end   = date.today()
default_start = default_end - timedelta(days=5 * 365)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    label_visibility="collapsed",
)
end_date = st.sidebar.date_input(
    "End Date",
    value=default_end,
    label_visibility="collapsed",
)

sidebar_section_divider()

# ── Monte Carlo Controls ──
sidebar_section_header("Monte Carlo Simulation", "🎲")
n_simulations = st.sidebar.slider(
    "Simulations",
    min_value=500,
    max_value=5000,
    value=1000,
    step=500,
    help="Number of simulation paths",
    label_visibility="collapsed",
)
mc_horizon = st.sidebar.slider(
    "Time Horizon (days)",
    min_value=30,
    max_value=252,
    value=252,
    step=30,
    help="Number of days to forecast",
    label_visibility="collapsed",
)

sidebar_section_divider()

# ── DCF Valuation Controls ──
sidebar_section_header("DCF Valuation", "💰")
forecast_yrs = st.sidebar.slider(
    "Forecast Period (years)",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Years to forecast",
    label_visibility="collapsed",
)
wacc = st.sidebar.slider(
    "WACC (%)",
    min_value=5.0,
    max_value=20.0,
    value=10.0,
    step=0.5,
    help="Weighted Average Cost of Capital",
    label_visibility="collapsed",
)
terminal_g = st.sidebar.slider(
    "Terminal Growth (%)",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.5,
    help="Long-term growth rate",
    label_visibility="collapsed",
)

sidebar_section_divider()

# ── Stress Testing Controls ──
sidebar_section_header("Stress Testing", "💥")
shock_scenario = st.sidebar.slider(
    "Price Shock (%)",
    min_value=-50.0,
    max_value=50.0,
    value=0.0,
    step=5.0,
    help="Apply custom price shock to portfolio",
    label_visibility="collapsed",
)

sidebar_section_divider()

# ── Data Control ──
sidebar_section_header("Data Management", "⚙️")
refresh = st.sidebar.button(
    "🔄 Refresh All Data",
    use_container_width=True,
    help="Clear cache and re-download latest data",
)

# ============================================================
# DATA FETCHING — Cached for performance
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance (cached for 1 hour)."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_tickers(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download Close prices for all tickers for portfolio analysis."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
    close.dropna(how="all", inplace=True)
    return close


# Clear cache on refresh
if refresh:
    fetch_stock_data.clear()
    fetch_all_tickers.clear()
    st.sidebar.success("✅ Cache cleared — refreshing data...")

# ── Fetch data with status indicators ──
status_col = st.sidebar.empty()

with st.spinner(f"📡 Fetching {selected_ticker}..."):
    df = fetch_stock_data(selected_ticker, str(start_date), str(end_date))

# ✅ Check if data is empty BEFORE using it
if df is None or df.empty:
    st.error("❌ No data available. Please check ticker or adjust date range.")
    st.stop()

with st.spinner("📊 Fetching portfolio data..."):
    all_close = fetch_all_tickers(TICKERS, str(start_date), str(end_date))

status_col.success("✅ Data loaded successfully")


# ============================================================
# HEADER — Premium Dashboard Header
# ============================================================

# Safe market open check — handles timezone-aware and naive DatetimeIndex
try:
    last_date = pd.Timestamp(df.index[-1]).date()
    market_open = last_date >= (default_end - timedelta(days=4))
except Exception:
    market_open = False

render_header(
    ticker=selected_ticker,
    start_date=str(start_date),
    end_date=str(end_date),
    market_open=market_open,
)


# ============================================================
# TAB NAVIGATION — Grouped & Professional
# ============================================================

tab_groups = {
    "Valuation & Forecasting": [
        ("📋", "Executive Summary", "summary"),
        ("📈", "ARIMA Forecast", "arima"),
        ("💰", "DCF Valuation", "dcf"),
        ("🎲", "Monte Carlo", "mc"),
        ("🌪️", "GARCH Volatility", "garch"),
    ],
    "Risk & Portfolio": [
        ("⚠️", "Value at Risk", "var"),
        ("🏦", "Credit Risk", "credit"),
        ("🗂️", "Portfolio", "portfolio"),
        ("💥", "Stress Testing", "stress"),
        ("🔥", "Correlation", "correlation"),
    ],
}

# Flatten tabs for rendering
all_tabs = []
tab_keys = []
for group, tabs_in_group in tab_groups.items():
    for icon, label, key in tabs_in_group:
        all_tabs.append(f"{icon} {label}")
        tab_keys.append(key)

# Render tabs with improved styling
st.markdown(
    """
    <style>
    .tab-group-header {
        color: #58a6ff;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(all_tabs)

# Prepare parameters for all modules
params = dict(
    df=df,
    ticker=selected_ticker,
    all_close=all_close,
    n_simulations=n_simulations,
    mc_horizon=mc_horizon,
    wacc=wacc / 100,
    terminal_g=terminal_g / 100,
    forecast_yrs=forecast_yrs,
    start_date=str(start_date),
    end_date=str(end_date),
    shock_scenario=shock_scenario / 100,
)


# ============================================================
# RENDER MODULES IN TABS
# ============================================================

# Group 1: Valuation & Forecasting
with tabs[0]:
    render_executive_summary(**params)

with tabs[1]:
    render_arima_forecast(**params)

with tabs[2]:
    render_dcf_valuation(**params)

with tabs[3]:
    render_monte_carlo(**params)

with tabs[4]:
    render_garch_volatility(**params)

# Group 2: Risk & Portfolio
with tabs[5]:
    render_value_at_risk(**params)

with tabs[6]:
    render_credit_risk(**params)

with tabs[7]:
    render_portfolio_optimization(**params)

with tabs[8]:
    render_stress_testing(**params)

with tabs[9]:
    render_correlation_heatmap(**params)


# ============================================================
# FOOTER — Professional Close
# ============================================================

section_divider()
render_footer()