# ============================================================
# modules/stress_testing.py — Module 9
# Portfolio Stress Testing — Scenario Analysis
# ============================================================
# Pre-defined macro-economic stress scenarios are applied
# to the equal-weight portfolio to estimate drawdown impact.
# A custom scenario slider lets users define their own shock.
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"

# Default scenarios: (label, equity_shock, bond_effect, notes)
BASE_SCENARIOS = [
    ("📉 Market Crash (-20%)",    -0.20,  0.05,
     "Broad market selloff, panic selling, liquidity crunch."),
    ("📈 Rate Hike (+200bps)",    -0.08, -0.10,
     "RBI hikes rates; bond prices fall, equities under pressure."),
    ("🌧️ Recession",             -0.15,  0.08,
     "GDP contraction, rising unemployment, credit tightening."),
    ("🛢️ Oil Shock (+25%)",       -0.10,  0.02,
     "Crude spike raises input costs; energy importers suffer."),
    ("🌟 Best Case (+15%)",        0.15,  0.03,
     "Strong earnings, policy support, global risk-on rally."),
]


def _apply_scenario(weights: np.ndarray, shock: float,
                    portfolio_value: float = 1_000_000) -> dict:
    """
    Apply a single equity shock to the portfolio.
    Gold/bond component assumed 10% of portfolio — opposite reaction.
    """
    equity_pct  = 0.90
    bond_pct    = 0.10
    bond_shock  = -shock * 0.3    # inverse but muted

    equity_impact = portfolio_value * equity_pct * shock
    bond_impact   = portfolio_value * bond_pct   * bond_shock
    total_impact  = equity_impact + bond_impact
    new_value     = portfolio_value + total_impact
    pct_change    = total_impact / portfolio_value * 100
    return {
        "Impact (₹)": total_impact,
        "New Value (₹)": new_value,
        "Change (%)": pct_change,
    }


def render_stress_testing(all_close, **kwargs):
    """Render Module 9 — Stress Testing."""

    st.subheader("💥 Stress Testing — Scenario Analysis")
    st.markdown(
        "Portfolio impact under predefined macro-economic scenarios. "
        "Add a **custom scenario** using the slider below."
    )

    # ── Equal weight from available tickers ──
    n      = all_close.shape[1]
    w      = np.ones(n) / n
    port_v = 1_000_000    # ₹10 lakh hypothetical portfolio

    # ── Custom Scenario ──
    st.markdown("#### ⚙️ Custom Scenario")
    custom_shock  = st.slider("Custom Equity Shock (%)", -50, 50, -10, step=1)
    custom_label  = st.text_input("Custom Scenario Name", value="My Scenario")
    custom_note   = f"User-defined equity shock of {custom_shock:+d}%."

    all_scenarios = BASE_SCENARIOS + [
        (custom_label, custom_shock / 100, 0, custom_note)
    ]

    # ── Compute impacts ──
    labels    = []
    impacts   = []
    new_vals  = []
    changes   = []
    notes_list= []

    for label, shock, _, note in all_scenarios:
        r = _apply_scenario(w, shock, port_v)
        labels.append(label)
        impacts.append(r["Impact (₹)"])
        new_vals.append(r["New Value (₹)"])
        changes.append(r["Change (%)"])
        notes_list.append(note)

    # ── Horizontal Bar Chart ──
    bar_colors = [GREEN if v >= 0 else RED for v in impacts]
    fig = go.Figure(go.Bar(
        x=impacts,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"₹{v:,.0f}  ({c:+.1f}%)" for v, c in zip(impacts, changes)],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color=ACCENT, line_dash="dash")
    fig.update_layout(
        title=f"Portfolio Stress Test Impact (Base: ₹{port_v:,.0f})",
        xaxis_title="Portfolio Impact (₹)",
        height=460,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=230, r=150, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario Table ──
    st.markdown("#### 📋 Detailed Results")
    result_df = pd.DataFrame({
        "Scenario":       labels,
        "Shock":          [f"{s*100:+.0f}%" for _, s, _, _ in all_scenarios],
        "Impact (₹)":    [f"₹{v:,.0f}"   for v in impacts],
        "New Value (₹)": [f"₹{v:,.0f}"   for v in new_vals],
        "Change (%)":    [f"{c:+.2f}%"   for c in changes],
    })
    st.dataframe(result_df.set_index("Scenario"), use_container_width=True)

    # ── Dynamic Interpretation ──
    worst_idx   = int(np.argmin(impacts))
    best_idx    = int(np.argmax(impacts))
    worst_label = labels[worst_idx]
    worst_pct   = changes[worst_idx]
    best_label  = labels[best_idx]
    best_pct    = changes[best_idx]

    st.markdown("---")
    st.markdown("#### 📝 Interpretation")
    st.info(
        f"**Worst scenario:** *{worst_label}* causes a portfolio drawdown of "
        f"**{worst_pct:.1f}%**, reducing ₹{port_v:,.0f} to "
        f"**₹{new_vals[worst_idx]:,.0f}**.\n\n"
        f"**Best scenario:** *{best_label}* grows the portfolio by "
        f"**{best_pct:+.1f}%** to **₹{new_vals[best_idx]:,.0f}**.\n\n"
        f"Scenario notes:\n"
        + "\n".join(f"- **{lbl}**: {note}"
                    for lbl, note in zip(labels, notes_list))
    )
