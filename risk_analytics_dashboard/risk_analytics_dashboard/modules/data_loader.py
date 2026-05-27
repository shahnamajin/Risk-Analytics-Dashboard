# =============================================================
# modules/data_loader.py — Shared data fetching utilities
# Uses yfinance with Streamlit caching to avoid repeated downloads
# =============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yfinance.
    Returns a clean DataFrame indexed by Date.
    Cached for 1 hour to avoid redundant API calls.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        st.error(f"No data returned for {ticker}. Check ticker symbol or date range.")
        return pd.DataFrame()
    df = df.dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_tickers(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download closing prices for multiple tickers.
    Returns a DataFrame where each column is a ticker's Close price.
    """
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()

    # Handle single vs multi-ticker column structure
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return close.dropna()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns: ln(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """Compute daily percentage returns: (P_t - P_{t-1}) / P_{t-1}"""
    return prices.pct_change().dropna()


def get_ticker_info(ticker: str) -> dict:
    """Fetch fundamental info dict from yfinance (sector, name, market cap, etc.)"""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}


def get_financials(ticker: str):
    """
    Fetch income statement, balance sheet, and cash flow from yfinance.
    Returns a tuple: (income_stmt, balance_sheet, cashflow)
    All are DataFrames. Returns None on failure.
    """
    try:
        t = yf.Ticker(ticker)
        return t.financials, t.balance_sheet, t.cashflow
    except Exception:
        return None, None, None


# ── Plotly Dark Theme Helper ──────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1220",
    font=dict(color="#e0e6f0", family="monospace"),
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=40),
)


def apply_dark_theme(fig):
    """Apply the standard dark finance theme to any Plotly figure."""
    fig.update_layout(**DARK_LAYOUT)
    return fig
