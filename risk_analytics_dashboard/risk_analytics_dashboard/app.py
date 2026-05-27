# ============================================================
# app.py — Main Entry Point
# Risk Analytics Dashboard | MCA Capstone Project
# ============================================================
# This is the central hub of the dashboard.
# It sets up the sidebar, imports each module, and
# renders tabs for every analytics section.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# ── Page config (MUST be the very first Streamlit call) ──
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-finance CSS theme ──
st.markdown(
    """
    <style>
    /* ── Background & base ── */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* ── Cards / metric boxes ── */
    div[data-testid="metric-container"] {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }

    /* ── Headings ── */
    h1, h2, h3 { color: #e6edf3; }

    /* ── Sidebar labels ── */
    .css-1d391kg p, label { color: #8b949e; }

    /* ── Plotly chart background override ── */
    .js-plotly-plot .plotly { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
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
# SIDEBAR — User Controls
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/stock-market.png", width=60
)
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

TICKERS = ["INFY.NS", "RELIANCE.NS", "ITC.NS", "HDFCBANK.NS", "ABB.NS"]

selected_ticker = st.sidebar.selectbox(
    "🏢 Select Stock", TICKERS, index=0,
    help="Choose the NSE stock to analyse across all modules."
)

default_end   = date.today()
default_start = default_end - timedelta(days=5 * 365)

start_date = st.sidebar.date_input("📅 Start Date", value=default_start)
end_date   = st.sidebar.date_input("📅 End Date",   value=default_end)

st.sidebar.markdown("---")
st.sidebar.markdown("### Monte Carlo")
n_simulations = st.sidebar.slider("Simulations", 500, 5000, 1000, step=500)
mc_horizon    = st.sidebar.slider("Horizon (days)", 30, 252, 252, step=30)

st.sidebar.markdown("### DCF Valuation")
wacc          = st.sidebar.slider("WACC (%)",            5.0, 20.0, 10.0, step=0.5)
terminal_g    = st.sidebar.slider("Terminal Growth (%)", 1.0,  8.0,  3.0, step=0.5)
forecast_yrs  = st.sidebar.slider("Forecast Years",       3,   10,    5)

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh All Data")


# ============================================================
# DATA FETCHING — cached so we don't re-download every render
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Returns a DataFrame with DatetimeIndex.
    ttl=3600 means cache expires after 1 hour.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_tickers(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download Close prices for all tickers (used by portfolio & heatmap modules).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex when multiple tickers; extract Close
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
    close.dropna(how="all", inplace=True)
    return close


# Clear cache when user clicks Refresh
if refresh:
    fetch_stock_data.clear()
    fetch_all_tickers.clear()
    st.success("✅ Cache cleared — data will re-download.")

# ── Fetch data ──
with st.spinner(f"Fetching data for {selected_ticker} …"):
    df = fetch_stock_data(selected_ticker, str(start_date), str(end_date))

with st.spinner("Fetching portfolio data …"):
    all_close = fetch_all_tickers(TICKERS, str(start_date), str(end_date))


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <h1 style='text-align:center; color:#58a6ff; margin-bottom:4px;'>
        📊 Risk Analytics Dashboard
    </h1>
    <p style='text-align:center; color:#8b949e; font-size:14px; margin-top:0;'>
        MCA Capstone Project &nbsp;|&nbsp; NSE/BSE Data &nbsp;|&nbsp;
        Selected: <b style='color:#f0883e;'>{selected_ticker}</b> &nbsp;|&nbsp;
        {start_date} → {end_date}
    </p>
    <hr style='border-color:#30363d;'>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.error("❌ No data returned. Check ticker symbol or date range.")
    st.stop()


# ============================================================
# TABS — one per module
# ============================================================
tabs = st.tabs([
    "📋 Executive Summary",
    "📈 ARIMA Forecast",
    "🌪️ GARCH Volatility",
    "💰 DCF Valuation",
    "🎲 Monte Carlo",
    "⚠️ Value at Risk",
    "🏦 Credit Risk",
    "🗂️ Portfolio Optimization",
    "💥 Stress Testing",
    "🔥 Correlation Heatmap",
])

params = dict(
    df=df,
    ticker=selected_ticker,
    all_close=all_close,
    n_simulations=n_simulations,
    mc_horizon=mc_horizon,
    wacc=wacc / 100,          # convert % → decimal
    terminal_g=terminal_g / 100,
    forecast_yrs=forecast_yrs,
    start_date=str(start_date),
    end_date=str(end_date),
)

with tabs[0]:  render_executive_summary(**params)
with tabs[1]:  render_arima_forecast(**params)
with tabs[2]:  render_garch_volatility(**params)
with tabs[3]:  render_dcf_valuation(**params)
with tabs[4]:  render_monte_carlo(**params)
with tabs[5]:  render_value_at_risk(**params)
with tabs[6]:  render_credit_risk(**params)
with tabs[7]:  render_portfolio_optimization(**params)
with tabs[8]:  render_stress_testing(**params)
with tabs[9]:  render_correlation_heatmap(**params)
