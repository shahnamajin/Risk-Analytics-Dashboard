# ============================================================
# DASHBOARD STYLING GUIDE
# How to Update Module Files for Professional Chart Styling
# ============================================================

## Overview

All module files should use the `chart_container` function from `ui_components.py`
to render Plotly charts with consistent professional styling.

## Import Required Components

```python
from .ui_components import (
    chart_container,
    section_divider,
    COLORS,
)
```

## Chart Container Function

The `chart_container(title, fig)` function:
- Applies consistent dark theme styling to Plotly figures
- Sets proper margins, gridlines, font colors
- Positions legend in top-right corner
- Uses unified hover mode
- Wraps chart in styled container div

## Standard Chart Styling Pattern

### Before (Old Style):
```python
st.markdown("#### 📈 My Chart Title")
fig = go.Figure()
fig.add_trace(...)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
)
st.plotly_chart(fig, use_container_width=True)
```

### After (New Style):
```python
from .ui_components import chart_container, COLORS

fig = go.Figure()
fig.add_trace(...)
# Don't set layout colors manually!
# chart_container handles it automatically

chart_container("My Chart Title", fig)
```

## Available Colors

Use from `COLORS` dict:
- `COLORS["accent_green"]` — ₹ Positive, gains
- `COLORS["accent_red"]` — ₹ Negative, losses
- `COLORS["accent_blue"]` — Primary, default
- `COLORS["accent_amber"]` — Warning, caution
- `COLORS["accent_purple"]` — Alternative accent

## Module Update Checklist

For each module file (`arima_forecast.py`, `garch_volatility.py`, etc.):

1. ✓ Add imports at top:
   ```python
   from .ui_components import chart_container, section_divider, COLORS
   ```

2. ✓ Replace color constants with COLORS dict:
   ```python
   # OLD: ACCENT = "#58a6ff"
   # NEW: Use COLORS["accent_blue"]
   ```

3. ✓ Add section dividers between major sections:
   ```python
   section_divider()
   ```

4. ✓ Wrap all Plotly charts with `chart_container()`:
   - Remove manual `st.markdown("#### Title")`
   - Remove `st.plotly_chart(fig, ...)`
   - Add `chart_container("Title", fig)`

5. ✓ Update Plotly figure layouts:
   - Remove `paper_bgcolor`, `plot_bgcolor` (handled by function)
   - Keep `trace` definitions and custom logic
   - Let `chart_container` set standard layout

## Example: Converting a Module

### BEFORE (old arima_forecast.py):
```python
st.markdown("#### 📈 ARIMA Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color=ACCENT)))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
)
st.plotly_chart(fig, use_container_width=True)
```

### AFTER (new arima_forecast.py):
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color=COLORS["accent_blue"])))
# Don't update layout for colors — chart_container does it!

chart_container("ARIMA Forecast", fig)
```

## Layout Recommendations

### Main Content Structure:
```python
st.markdown("### 📊 Section Title")
st.markdown("Brief description")

# Charts
chart_container("Chart Title 1", fig1)
section_divider()
chart_container("Chart Title 2", fig2)
section_divider()

# Metrics
cols = st.columns(4)
with cols[0]:
    st.metric("Label", "Value", "Delta")
```

## Sidebar Usage

Use `sidebar_section_header()` and `sidebar_section_divider()`:
```python
from .ui_components import sidebar_section_header, sidebar_section_divider

sidebar_section_header("Controls", "⚙️")
# Add controls here
sidebar_section_divider()
```

## KPI Cards

Use the HTML-based card styling:
```python
st.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Label</div>
        <div class="kpi-value">Value</div>
        <div class="kpi-trend">Trend Info</div>
    </div>
    """,
    unsafe_allow_html=True,
)
```

## Responsive Design

The CSS automatically handles:
- Mobile (< 768px)
- Tablet (768px - 1024px)
- Desktop (> 1024px)

Use `st.columns()` for layouts — they auto-adjust.

## Color Semantic Meanings

- **Green** (#3fb950): Positive, growth, low risk
- **Red** (#f85149): Negative, decline, high risk
- **Amber** (#d29922): Warning, medium risk, caution
- **Blue** (#58a6ff): Neutral, primary, default
- **Purple** (#bc8ef9): Alternative, secondary accent

## Testing Changes

1. Update one module completely
2. Run: `streamlit run app.py`
3. Navigate to that module's tab
4. Verify:
   - ✓ Charts render properly
   - ✓ Colors are consistent
   - ✓ Text is readable
   - ✓ Margins/spacing look good
   - ✓ Hover tooltips work
5. Then update next module

## Common Issues & Fixes

### Issue: Chart looks broken
**Fix:** Make sure you're using `chart_container()`, not `st.plotly_chart()`

### Issue: Colors don't match
**Fix:** Use `COLORS` dict instead of hardcoded hex values

### Issue: Legend in wrong position
**Fix:** Don't set `legend` in figure layout — `chart_container` handles it

### Issue: Text unreadable
**Fix:** Check that font size is >= 10, use COLORS["text_primary"] for text

## Module Files to Update

1. ✓ `executive_summary.py` — DONE
2. ⏳ `arima_forecast.py`
3. ⏳ `garch_volatility.py`
4. ⏳ `dcf_valuation.py`
5. ⏳ `monte_carlo.py`
6. ⏳ `value_at_risk.py`
7. ⏳ `credit_risk.py`
8. ⏳ `portfolio_optimization.py`
9. ⏳ `stress_testing.py`
10. ⏳ `correlation_heatmap.py`

## Next Steps

Apply this pattern to each module systematically for complete UI consistency.
