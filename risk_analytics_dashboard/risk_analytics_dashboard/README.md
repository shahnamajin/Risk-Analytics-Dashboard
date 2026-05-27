# 📊 Risk Analytics Dashboard
### MCA Capstone Project | Python · Streamlit · Plotly · NSE/BSE Data

> An end-to-end **professional financial risk analytics dashboard** built with Python,
> powered by real NSE/BSE stock data from Yahoo Finance.

---

## 🗂️ Project Structure

```
risk_analytics_dashboard/
│
├── app.py                        ← Main Streamlit entry point
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── data/                         ← (reserved for local CSV caching)
│
└── modules/
    ├── __init__.py
    ├── executive_summary.py      ← Module 1: KPI cards + signal
    ├── arima_forecast.py         ← Module 2: ARIMA price forecast
    ├── garch_volatility.py       ← Module 3: GARCH(1,1) volatility
    ├── dcf_valuation.py          ← Module 4: DCF intrinsic value
    ├── monte_carlo.py            ← Module 5: GBM Monte Carlo
    ├── value_at_risk.py          ← Module 6: VaR / CVaR / Kupiec
    ├── credit_risk.py            ← Module 7: Logistic Regression PD
    ├── portfolio_optimization.py ← Module 8: Efficient Frontier
    ├── stress_testing.py         ← Module 9: Scenario analysis
    └── correlation_heatmap.py    ← Module 10: Return correlations
```

---

## 🚀 Quick Start

### 1. Clone / download the project
```bash
git clone https://github.com/YOUR_USERNAME/risk_analytics_dashboard.git
cd risk_analytics_dashboard
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** in your browser.

---

## 📦 Tech Stack

| Library         | Purpose                                      |
|-----------------|----------------------------------------------|
| `streamlit`     | Interactive web dashboard                    |
| `plotly`        | Professional interactive charts              |
| `yfinance`      | Real NSE/BSE stock data                      |
| `pandas`        | Data manipulation                            |
| `numpy`         | Numerical computation                        |
| `pmdarima`      | Auto-ARIMA model selection                   |
| `arch`          | GARCH volatility modelling                   |
| `statsmodels`   | Statistical tools                            |
| `scikit-learn`  | Logistic Regression for credit risk          |
| `scipy`         | Statistical distributions, Kupiec test       |
| `PyPortfolioOpt`| Portfolio optimization helpers               |

---

## 📋 Module Overview

| # | Module | Key Technique |
|---|--------|---------------|
| 1 | Executive Summary | KPI aggregation, BUY/HOLD/SELL signal |
| 2 | ARIMA Forecast | auto_arima, walk-forward validation |
| 3 | GARCH Volatility | GARCH(1,1), regime detection |
| 4 | DCF Valuation | FCF projection, terminal value |
| 5 | Monte Carlo | GBM, 1000+ paths, probability stats |
| 6 | Value at Risk | Historical / Parametric / MC VaR, Kupiec |
| 7 | Credit Risk | Logistic Regression, ROC, confusion matrix |
| 8 | Portfolio Optimization | Efficient Frontier, Max Sharpe |
| 9 | Stress Testing | Macro scenarios, custom shock slider |
| 10 | Correlation Heatmap | Pearson correlation, rolling corr |

---

## ⚙️ Sidebar Controls

| Control | Effect |
|---------|--------|
| Ticker selector | Changes the stock analysed in all modules |
| Date pickers | Adjusts historical data range |
| Simulations slider | Monte Carlo simulation count |
| Horizon slider | Forecast horizon (days) |
| WACC slider | Discount rate for DCF |
| Terminal Growth | DCF terminal growth rate |
| Forecast Years | DCF projection period |
| 🔄 Refresh | Clears cache and re-downloads data |

---

## 📈 Stocks Covered

| Ticker | Company |
|--------|---------|
| `INFY.NS` | Infosys Ltd |
| `RELIANCE.NS` | Reliance Industries |
| `ITC.NS` | ITC Ltd |
| `HDFCBANK.NS` | HDFC Bank |
| `ABB.NS` | ABB India Ltd |

---

## 🎓 Educational Notes

This project is designed to be **beginner-friendly** while covering **professional-grade** risk models:

- Every function has **docstrings** explaining what it does.
- Inline comments explain the **financial intuition** behind formulas.
- Models use **sensible defaults** that work out of the box.
- The DCF module uses **real cashflow data** from yfinance with graceful fallbacks.
- The Credit Risk module uses **synthetic data** to avoid overfitting to real labels.

---

## 🐛 Common Issues

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| ARIMA tab is slow | First load fits the model (~15 s); subsequent loads use cache |
| Empty charts | Check internet connection; yfinance requires it |
| `pmdarima` install fails | Try `pip install pmdarima --no-build-isolation` |

---

## 📄 License

MIT License — free to use for academic and personal projects.

---

*Built as MCA Capstone Project | Risk Analytics Dashboard | Python + Streamlit*
