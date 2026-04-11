"""
Restaurant Rating Dashboard — driven by Dataset.csv
Run: streamlit run rating_dashboard.py
Requires: streamlit plotly pandas
Place Dataset.csv in the same directory as this script.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Rating Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f5f3ef; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"]  { font-size: 11px !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.6px; color: #6b6860 !important; }
[data-testid="stMetricValue"]  { font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.8px; color: #1a1917 !important; }
[data-testid="stMetricDelta"]  { font-size: 11px !important; font-family: 'DM Mono', monospace; }

.card { background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05); margin-bottom: 16px; }
.section-title { font-size: 13px; font-weight: 600; color: #1a1917; letter-spacing: -0.1px; margin-bottom: 14px; }

.dashboard-header { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 28px; }
.header-title  { font-size: 22px; font-weight: 600; letter-spacing: -0.3px; color: #1a1917; }
.header-sub    { font-size: 13px; color: #6b6860; margin-top: 2px; font-family: 'DM Mono', monospace; }
.header-badge  { font-family: 'DM Mono', monospace; font-size: 11px; background: #ffffff; border: 1px solid rgba(0,0,0,0.08); color: #6b6860; padding: 5px 10px; border-radius: 999px; display: inline-block; }

.insight { background: #fef9ec; border-left: 3px solid #d97706; border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 12px; color: #7a4f0a; margin-top: 14px; line-height: 1.5; }

.bar-row-label { display: flex; justify-content: space-between; font-size: 12px; color: #6b6860; margin-bottom: 4px; }
.bar-row-label .lbl { font-weight: 500; color: #1a1917; }
.bar-row-label .cnt { font-family: 'DM Mono', monospace; font-size: 11px; }
.bar-track { height: 8px; background: #f0ede8; border-radius: 999px; overflow: hidden; margin-bottom: 9px; }
.bar-fill  { height: 100%; border-radius: 999px; }

.legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #6b6860; }
.legend-dot  { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""",unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C = dict(
    none="#888780", poor="#c0392b", avg="#d97706",
    good="#5a8a1a", vgood="#0e7c55", excel="#064e38",
    blue="#2563b0", muted="#9c9a94", grid="rgba(0,0,0,0.06)",
)
FONT      = dict(family="DM Sans, sans-serif", color="#1a1917")
TICK_FONT = dict(family="DM Mono, monospace", size=10, color=C["muted"])

def base_layout(height=220):
    return dict(
        height=height, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=FONT, showlegend=False,
    )

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path  / "processed_data" / "Dataset_filtered.csv"
    
    if file_path.exists():
        return pd.read_csv(file_path)
    st.error(f"Dataset not found at: {file_path}")
    st.stop()

df = load_data()

# ── Derived stats ─────────────────────────────────────────────────────────────
total      = len(df)
not_rated  = int((df["Aggregate rating"] == 0).sum())
rated_df   = df[df["Aggregate rating"] > 0]
mean_all   = round(df["Aggregate rating"].mean(), 2)
mean_rated = round(rated_df["Aggregate rating"].mean(), 2)
pct_not    = round(not_rated / total * 100, 1)

rating_counts = df["Rating text"].value_counts()
def rc(label): return int(rating_counts.get(label, 0))

poor_n  = rc("Poor");  avg_n  = rc("Average"); good_n = rc("Good")
vgood_n = rc("Very Good"); excel_n = rc("Excellent")
rated_n = total - not_rated

def pct(n): return round(n / total * 100, 1)

imbalance       = round((avg_n / rated_n * 100) / (excel_n / rated_n * 100), 1) if excel_n else "N/A"
avg_pct_rated   = round(avg_n  / rated_n * 100, 1)
excel_pct_rated = round(excel_n / rated_n * 100, 1)

price_counts    = df["Price range"].value_counts().sort_index()
delivery_counts = df["Has Online delivery"].value_counts()
booking_counts  = df["Has Table booking"].value_counts()

# Histogram
bin_edges = [0, 1.85, 2.05, 2.25, 2.45, 2.65, 2.85, 3.05, 3.25, 3.45, 3.65, 3.85, 4.05, 4.25, 4.45, 4.65, 4.85, 5.0]
hist_bins = ['0','1.8','2.0','2.2','2.4','2.6','2.8','3.0','3.2','3.4','3.6','3.8','4.0','4.2','4.4','4.6','4.8']
hist_vals = []
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    if i == 0:
        cnt = int((df["Aggregate rating"] == 0).sum())
    else:
        cnt = int(((df["Aggregate rating"] >= lo) & (df["Aggregate rating"] < hi)).sum())
    hist_vals.append(cnt)

hist_colors = [
    C["none"],
    C["poor"], C["poor"], C["poor"], C["poor"],
    C["avg"],  C["avg"],  C["avg"],  C["avg"],  C["avg"],
    C["good"], C["good"],
    C["vgood"],C["vgood"],C["vgood"],
    C["excel"],C["excel"],
]

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dashboard-header">
  <div>
    <div class="header-title">Restaurant Rating Dashboard</div>
    <div class="header-sub">Restaurant Dataset · {total:,} restaurants · Aggregate rating analysis</div>
  </div>
  <div class="header-badge">EDA · Class Imbalance</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — Metric cards
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Restaurants", f"{total:,}")
with c2: st.metric("Not Rated (0)", f"{not_rated:,}", delta=f"{pct_not}% of dataset", delta_color="inverse")
with c3: st.metric("Mean (all)",       f"{mean_all}",   help="Includes zero ratings")
with c4: st.metric("Mean (rated only)", f"{mean_rated}", help="Excludes zero ratings")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — Class distribution + Donut
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns(2)

with left:
    def bar_row(label, count, color):
        p = pct(count)
        return f"""
        <div class="bar-row-label"><span class="lbl">{label}</span><span class="cnt">{count:,} ({p}%)</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{p}%;background:{color}"></div></div>
        """
    st.markdown(f"""
        <div class="card">
        <div class="section-title">Rating class distribution</div>
        {bar_row("Not rated", not_rated, C["none"])}
        {bar_row("Poor (1.8–2.4)", poor_n, C["poor"])}
        {bar_row("Average (2.5–3.4)", avg_n, C["avg"])}
        {bar_row("Good (3.5–3.9)", good_n, C["good"])}
        {bar_row("Very Good (4.0–4.4)", vgood_n, C["vgood"])}
        {bar_row("Excellent (4.5–4.9)", excel_n, C["excel"])}
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="section-title">Rated vs not rated</div>', unsafe_allow_html=True)
    donut = go.Figure(go.Pie(
        labels=[f"Rated ({100-pct_not}%)", f"Not Rated ({pct_not}%)"],
        values=[rated_n, not_rated],
        hole=0.62,
        marker=dict(colors=[C["blue"], C["none"]], line=dict(color="#fff", width=3)),
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    layout = base_layout(260)
    layout["showlegend"] = True

    donut.update_layout(
        **layout,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.05,
            font=dict(family="DM Sans, sans-serif", size=12, color=C["muted"])
        ),
    )
    st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 3 — Histogram
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="card"><div class="section-title">Aggregate rating histogram (all {total:,} entries)</div>', unsafe_allow_html=True)
hist_fig = go.Figure(go.Bar(
    x=hist_bins, y=hist_vals,
    marker=dict(color=hist_colors, line=dict(width=0)),
    hovertemplate="Rating %{x} — Count: %{y:,}<extra></extra>",
))
hist_fig.update_layout(
    **base_layout(240), bargap=0.12,
    xaxis=dict(tickfont=TICK_FONT, showgrid=False, linecolor=C["grid"], linewidth=1),
    yaxis=dict(tickfont=TICK_FONT, gridcolor=C["grid"], showline=False),
)
st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("""
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#888780"></div>Not rated (0)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#c0392b"></div>Poor</div>
  <div class="legend-item"><div class="legend-dot" style="background:#d97706"></div>Average</div>
  <div class="legend-item"><div class="legend-dot" style="background:#5a8a1a"></div>Good</div>
  <div class="legend-item"><div class="legend-dot" style="background:#0e7c55"></div>Very Good</div>
  <div class="legend-item"><div class="legend-dot" style="background:#064e38"></div>Excellent</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 4 — Secondary imbalances
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)

def small_bar(labels, values, colors, height=170):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(height), bargap=0.25,
        xaxis=dict(tickfont=TICK_FONT, showgrid=False, linecolor=C["grid"], tickangle=0),
        yaxis=dict(tickfont=TICK_FONT, gridcolor=C["grid"], showline=False),
    )
    return fig

with p1:
    st.markdown('<div class="card"><div class="section-title">Price range</div>', unsafe_allow_html=True)
    values = [int(price_counts.get(i, 0)) for i in [1, 2, 3, 4]]
    st.plotly_chart(small_bar(["Budget (1)","Mid (2)","Premium (3)","Luxury (4)"], values,
                               [C["blue"],C["vgood"],C["avg"],C["poor"]]),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="card"><div class="section-title">Online delivery</div>', unsafe_allow_html=True)
    values = [int(delivery_counts.get("No", 0)), int(delivery_counts.get("Yes", 0))]
    st.plotly_chart(small_bar(["No","Yes"], values, [C["poor"],C["blue"]]),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with p3:
    st.markdown('<div class="card"><div class="section-title">Table booking</div>', unsafe_allow_html=True)
    values = [int(booking_counts.get("No", 0)), int(booking_counts.get("Yes", 0))]
    st.plotly_chart(small_bar(["No","Yes"], values, [C["poor"],C["blue"]]),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
