# ============================================================
# modules/ui_components.py — Reusable UI Components
# All styles use INLINE CSS only — no external CSS class names.
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from textwrap import dedent

COLORS = {
    "primary_dark":   "#0d1117",
    "secondary_dark": "#161b22",
    "tertiary_dark":  "#1c2128",
    "border_subtle":  "#30363d",
    "text_primary":   "#e6edf3",
    "text_secondary": "#8b949e",
    "accent_blue":    "#58a6ff",
    "accent_green":   "#3fb950",
    "accent_red":     "#f85149",
    "accent_amber":   "#d29922",
    "accent_purple":  "#bc8ef9",
}


def render_html(html: str, target=st) -> None:
    target.markdown(dedent(html).strip(), unsafe_allow_html=True)


# ── KPI Card ────────────────────────────────────────────────
def kpi_card(label, value, trend=None, unit="%", color="blue", sparkline_data=None):
    color_map = {
        "green": COLORS["accent_green"], "red": COLORS["accent_red"],
        "blue":  COLORS["accent_blue"],  "amber": COLORS["accent_amber"],
        "neutral": COLORS["text_secondary"],
    }
    val_color  = color_map.get(color, COLORS["accent_blue"])
    trend_color = COLORS["accent_green"] if (trend is not None and trend >= 0) else COLORS["accent_red"]
    arrow = "↑" if (trend is not None and trend >= 0) else "↓"
    trend_html = (
        f'<div style="margin-top:6px;color:{trend_color};font-size:.84rem;font-weight:600;">'
        f'{arrow} {abs(trend):+.2f}{unit}</div>'
    ) if trend is not None else ""

    return (
        f'<div style="background:{COLORS["tertiary_dark"]};border:1px solid {COLORS["border_subtle"]};'
        f'border-radius:8px;padding:16px 18px;margin-bottom:8px;">'
        f'<div style="color:{COLORS["text_secondary"]};font-size:.76rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{val_color};font-size:1.35rem;font-weight:700;">{value}</div>'
        f'{trend_html}</div>'
    )


def render_kpi_grid(cards_data, cols=3):
    col_list = st.columns(cols)
    for idx, (label, (value, trend, color)) in enumerate(cards_data.items()):
        with col_list[idx % cols]:
            render_html(kpi_card(label, value, trend, "%", color))


# ── Investment Signal Box ────────────────────────────────────
def investment_signal(signal, confidence, risk_level, explanation):
    """
    Render BUY/HOLD/SELL signal box with fully inline styles.
    explanation must be a plain string (no embedded newlines or HTML tags).
    """
    cfg = {
        "BUY":  ("🟢 BUY",  COLORS["accent_green"], "rgba(63,185,80,0.12)"),
        "HOLD": ("🟡 HOLD", COLORS["accent_amber"], "rgba(210,153,34,0.12)"),
        "SELL": ("🔴 SELL", COLORS["accent_red"],   "rgba(248,81,73,0.12)"),
    }
    label, border, bg = cfg.get(signal, ("⚪ NEUTRAL", COLORS["text_secondary"], "rgba(139,148,158,0.12)"))
    risk_color = {"Low": COLORS["accent_green"], "Medium": COLORS["accent_amber"],
                  "High": COLORS["accent_red"]}.get(risk_level, COLORS["text_secondary"])

    # Sanitise explanation — strip newlines/extra spaces
    clean_exp = " ".join(str(explanation).split())

    html = (
        f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
        f'padding:24px;text-align:center;margin:12px 0;">'
        f'<div style="font-size:1.9rem;font-weight:800;color:{border};'
        f'letter-spacing:.08em;margin-bottom:16px;">{label}</div>'
        f'<div style="display:flex;justify-content:center;gap:48px;margin-bottom:16px;">'
        f'<div><div style="color:{COLORS["text_secondary"]};font-size:.76rem;'
        f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Confidence</div>'
        f'<div style="color:{COLORS["text_primary"]};font-size:1.3rem;font-weight:700;">'
        f'{confidence:.0f}%</div></div>'
        f'<div><div style="color:{COLORS["text_secondary"]};font-size:.76rem;'
        f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Risk Level</div>'
        f'<div style="color:{risk_color};font-size:1.3rem;font-weight:700;">'
        f'{risk_level}</div></div></div>'
        f'<div style="color:{COLORS["text_secondary"]};font-size:.87rem;line-height:1.5;'
        f'border-top:1px solid {COLORS["border_subtle"]};padding-top:12px;">'
        f'{clean_exp}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Chart Container ──────────────────────────────────────────
def chart_container(title, fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial,sans-serif", size=11, color=COLORS["text_primary"]),
        title=dict(text=title, x=0.02, y=0.98, xanchor="left", yanchor="top",
                   font=dict(size=14, color=COLORS["text_primary"])),
        hovermode="x unified", margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(48,54,61,.3)",
                   zeroline=False, linecolor=COLORS["border_subtle"], mirror=True),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(48,54,61,.3)",
                   zeroline=False, linecolor=COLORS["border_subtle"], mirror=True),
        legend=dict(orientation="v", yanchor="top", y=.99, xanchor="right", x=.99,
                    bgcolor="rgba(13,17,23,.8)", bordercolor=COLORS["border_subtle"], borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section Divider ──────────────────────────────────────────
def section_divider():
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {COLORS["border_subtle"]};margin:24px 0;">',
        unsafe_allow_html=True,
    )


# ── Sidebar Helpers ──────────────────────────────────────────
def sidebar_section_header(title, icon="⚙️"):
    st.sidebar.markdown(
        f'<div style="font-size:.76rem;font-weight:700;color:{COLORS["text_secondary"]};'
        f'text-transform:uppercase;letter-spacing:.08em;margin:16px 0 8px;">'
        f'{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def sidebar_section_divider():
    st.sidebar.markdown(
        f'<hr style="border:none;border-top:1px solid {COLORS["border_subtle"]};margin:10px 0;">',
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────
def render_header(ticker, start_date, end_date, market_open=True):
    sc = COLORS["accent_green"] if market_open else COLORS["accent_red"]
    st_text = "MARKET OPEN" if market_open else "MARKET CLOSED"
    st.markdown(
        f'<div style="background:{COLORS["secondary_dark"]};border:1px solid {COLORS["border_subtle"]};'
        f'border-radius:12px;padding:24px 28px;margin-bottom:20px;">'
        f'<h1 style="margin:0;font-size:1.75rem;font-weight:800;color:{COLORS["text_primary"]};">'
        f'📊 Risk Analytics Dashboard</h1>'
        f'<p style="margin:6px 0 0;color:{COLORS["text_secondary"]};font-size:.88rem;">'
        f'MCA Financial Analytics Capstone Project</p>'
        f'<div style="margin-top:10px;display:flex;gap:16px;flex-wrap:wrap;">'
        f'<span style="color:{COLORS["text_secondary"]};font-size:.8rem;">NSE/BSE Live Data</span>'
        f'<span style="color:{COLORS["accent_blue"]};font-size:.8rem;font-weight:600;">Selected: {ticker}</span>'
        f'<span style="color:{COLORS["text_secondary"]};font-size:.8rem;">{start_date} → {end_date}</span>'
        f'</div>'
        f'<div style="margin-top:12px;display:inline-block;background:rgba(63,185,80,.12);'
        f'border:1px solid {sc};border-radius:20px;padding:4px 12px;font-size:.78rem;'
        f'font-weight:700;color:{sc};">● {st_text}</div></div>',
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────
def render_footer():
    yr = datetime.now().year
    st.markdown(
        f'<div style="text-align:center;padding:24px 0;margin-top:32px;'
        f'border-top:1px solid {COLORS["border_subtle"]};">'
        f'<div style="color:{COLORS["text_secondary"]};font-size:.84rem;">'
        f'Built with Python &nbsp;•&nbsp; Streamlit &nbsp;•&nbsp; Plotly</div>'
        f'<div style="color:{COLORS["text_secondary"]};font-size:.8rem;margin-top:4px;">'
        f'MCA Financial Analytics Capstone Project — Data Source: Yahoo Finance</div>'
        f'<div style="color:#6e7681;font-size:.74rem;margin-top:6px;">'
        f'© {yr} Risk Analytics Dashboard.</div></div>',
        unsafe_allow_html=True,
    )


# ── Metric Badge ─────────────────────────────────────────────
def metric_badge(label, value, color="blue"):
    cm = {
        "green": (COLORS["accent_green"], "rgba(63,185,80,.15)"),
        "red":   (COLORS["accent_red"],   "rgba(248,81,73,.15)"),
        "blue":  (COLORS["accent_blue"],  "rgba(88,166,255,.15)"),
        "amber": (COLORS["accent_amber"], "rgba(210,153,34,.15)"),
    }
    tc, bg = cm.get(color, cm["blue"])
    return (
        f'<span style="display:inline-block;background:{bg};color:{tc};'
        f'padding:.35rem .75rem;border-radius:6px;font-size:.83rem;font-weight:600;'
        f'border:1px solid {tc}33;">{label}: <strong>{value}</strong></span>'
    )