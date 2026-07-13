import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="AirCast — Passenger Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# Global CSS — Cockpit Dark Theme
# ------------------------------------------------

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A1628 !important;
    color: #E8EDF4 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #0A1628 !important;
    padding: 0 !important;
}

/* ── Sidebar — Control Panel ── */
[data-testid="stSidebar"] {
    background-color: #071020 !important;
    border-right: 1px solid #1A3254 !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

.sidebar-header {
    background: linear-gradient(135deg, #112240 0%, #0A1628 100%);
    border-bottom: 1px solid #1A3254;
    padding: 28px 24px 22px;
    margin-bottom: 0;
}

.sidebar-header .brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #E8EDF4;
    letter-spacing: -0.5px;
    line-height: 1;
}

.sidebar-header .brand span {
    color: #1E6FD9;
}

.sidebar-header .tagline {
    font-size: 11px;
    color: #6B8CAE;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.sidebar-section {
    padding: 20px 24px;
    border-bottom: 1px solid #1A3254;
}

.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6B8CAE;
    margin-bottom: 14px;
}

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1A3254;
}

.sidebar-stat:last-child { border-bottom: none; }

.sidebar-stat .stat-name {
    font-size: 12px;
    color: #8BA8C4;
}

.sidebar-stat .stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #1E6FD9;
}

/* ── Slider overrides ── */
[data-testid="stSlider"] > div > div > div {
    background: #1A3254 !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #1E6FD9 !important;
}
[data-testid="stSlider"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    color: #8BA8C4 !important;
}

/* ── Main content padding ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Page header ── */
.page-hero {
    padding: 32px 0 28px;
    border-bottom: 1px solid #1A3254;
    margin-bottom: 32px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
}

.page-hero-left .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #1E6FD9;
    margin-bottom: 10px;
}

.page-hero-left h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 40px !important;
    font-weight: 700 !important;
    color: #E8EDF4 !important;
    letter-spacing: -1.2px !important;
    line-height: 1.1 !important;
    margin: 0 0 8px !important;
    padding: 0 !important;
}

.page-hero-left .subtitle {
    font-size: 14px;
    color: #6B8CAE;
    line-height: 1.5;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(30, 111, 217, 0.12);
    border: 1px solid rgba(30, 111, 217, 0.3);
    border-radius: 100px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #1E6FD9;
    white-space: nowrap;
}

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #1E6FD9;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6B8CAE;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1A3254;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: #E8EDF4;
    letter-spacing: -0.4px;
    margin: 0 0 24px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

.metric-card {
    background: #112240;
    border: 1px solid #1A3254;
    border-radius: 12px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: #1E6FD9; }

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1E6FD9, transparent);
}

.metric-card .mc-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6B8CAE;
    margin-bottom: 14px;
}

.metric-card .mc-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #E8EDF4;
    letter-spacing: -1px;
    line-height: 1;
}

.metric-card .mc-desc {
    font-size: 11px;
    color: #6B8CAE;
    margin-top: 8px;
}

/* ── Data panel ── */
.data-panel {
    background: #112240;
    border: 1px solid #1A3254;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 28px;
}

.data-panel-header {
    padding: 18px 24px;
    border-bottom: 1px solid #1A3254;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.data-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #C8D8E8;
}

.data-panel-body {
    padding: 16px 24px 20px;
}

/* ── Dataframe overrides ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stDataFrame"] iframe {
    border-radius: 8px !important;
}

/* ── Forecast button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1E6FD9 0%, #1558B0 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 28px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.2px !important;
    width: 100% !important;
    height: auto !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(30, 111, 217, 0.3) !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(30, 111, 217, 0.45) !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Forecast result cards ── */
.forecast-banner {
    background: linear-gradient(135deg, rgba(30,111,217,0.15) 0%, rgba(244,168,50,0.08) 100%);
    border: 1px solid rgba(30,111,217,0.3);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.forecast-banner .fb-icon {
    font-size: 28px;
    line-height: 1;
}

.forecast-banner .fb-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 500;
    color: #C8D8E8;
}

.forecast-banner .fb-text span {
    color: #1E6FD9;
    font-weight: 700;
}

/* ── Runway divider ── */
.runway {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 36px 0;
    height: 20px;
    opacity: 0.5;
}

.runway-segment {
    flex: 1;
    height: 3px;
    background: #1E6FD9;
    margin: 0 3px;
    border-radius: 2px;
}

.runway-segment:nth-child(even) {
    background: transparent;
}

/* ── Success / info messages ── */
[data-testid="stAlert"] {
    background: rgba(30, 111, 217, 0.1) !important;
    border: 1px solid rgba(30, 111, 217, 0.3) !important;
    border-radius: 8px !important;
    color: #8BBCE8 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #1E6FD9 !important; }

/* ── Download button ── */
div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #1A3254 !important;
    color: #8BA8C4 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    border-color: #1E6FD9 !important;
    color: #E8EDF4 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Tab overrides (hidden — we're using custom layout) ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1A3254 !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #6B8CAE !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    border-radius: 0 !important;
}

[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
    color: #E8EDF4 !important;
    border-bottom-color: #1E6FD9 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ── Column gaps ── */
[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Data
# ------------------------------------------------

loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()

# ------------------------------------------------
# Sidebar — Control Panel
# ------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="brand">Air<span>Cast</span></div>
        <div class="tagline">Passenger Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section">
        <div class="sidebar-section-label">Forecast Controls</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding: 16px 24px 8px;'>", unsafe_allow_html=True)
        future_months = st.slider(
            "Horizon (months)",
            min_value=1, max_value=24, value=12
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-section-label">Dataset</div>
        <div class="sidebar-stat">
            <span class="stat-name">Records</span>
            <span class="stat-val">{len(df)}</span>
        </div>
        <div class="sidebar-stat">
            <span class="stat-name">Start</span>
            <span class="stat-val">{df.index[0].strftime('%b %Y')}</span>
        </div>
        <div class="sidebar-stat">
            <span class="stat-name">End</span>
            <span class="stat-val">{df.index[-1].strftime('%b %Y')}</span>
        </div>
        <div class="sidebar-stat">
            <span class="stat-name">Peak</span>
            <span class="stat-val">{int(df['Passengers'].max()):,}</span>
        </div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-label">Model</div>
        <div class="sidebar-stat">
            <span class="stat-name">Architecture</span>
            <span class="stat-val">RNN</span>
        </div>
        <div class="sidebar-stat">
            <span class="stat-name">Window</span>
            <span class="stat-val">{future_months}mo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Hero Header
# ------------------------------------------------

st.markdown(f"""
<div class="page-hero">
    <div class="page-hero-left">
        <div class="eyebrow">✈ &nbsp;Temporal Demand Forecasting</div>
        <h1>Airline Passenger<br>Analysis</h1>
        <div class="subtitle">Historical trends · RNN-powered predictions · {future_months}-month horizon</div>
    </div>
    <div>
        <div class="status-pill">
            <div class="status-dot"></div>
            Model Ready
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Model Accuracy Section
# ------------------------------------------------

st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

mae, mse, rmse = Evaluator().evaluate()

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="mc-label">MAE</div>
        <div class="mc-value">{mae:.2f}</div>
        <div class="mc-desc">Mean Absolute Error</div>
    </div>
    <div class="metric-card">
        <div class="mc-label">MSE</div>
        <div class="mc-value">{mse:.2f}</div>
        <div class="mc-desc">Mean Squared Error</div>
    </div>
    <div class="metric-card">
        <div class="mc-label">RMSE</div>
        <div class="mc-value">{rmse:.2f}</div>
        <div class="mc-desc">Root Mean Squared Error</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Runway Divider
# ------------------------------------------------

segments = "".join(['<div class="runway-segment"></div>' for _ in range(40)])
st.markdown(f'<div class="runway">{segments}</div>', unsafe_allow_html=True)

# ------------------------------------------------
# EDA Section
# ------------------------------------------------

st.markdown('<div class="section-label">Exploratory Data Analysis</div>', unsafe_allow_html=True)

col_table, col_chart = st.columns([1, 2])

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8BA8C4", size=12),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(
        gridcolor="#1A3254",
        linecolor="#1A3254",
        tickcolor="#1A3254",
    ),
    yaxis=dict(
        gridcolor="#1A3254",
        linecolor="#1A3254",
        tickcolor="#1A3254",
    ),
)

with col_table:
    st.markdown('<div class="data-panel"><div class="data-panel-header"><span class="data-panel-title">Raw Dataset</span></div><div class="data-panel-body">', unsafe_allow_html=True)
    st.dataframe(
        df,
        height=340,
        use_container_width=True,
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

with col_chart:
    st.markdown('<div class="data-panel"><div class="data-panel-header"><span class="data-panel-title">Historical Passenger Volume</span></div><div class="data-panel-body">', unsafe_allow_html=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df.index,
        y=df["Passengers"],
        name="Passengers",
        line=dict(color="#1E6FD9", width=2),
        fill="tozeroy",
        fillcolor="rgba(30, 111, 217, 0.08)",
    ))
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=310)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# ------------------------------------------------
# Runway Divider
# ------------------------------------------------

st.markdown(f'<div class="runway">{segments}</div>', unsafe_allow_html=True)

# ------------------------------------------------
# Forecast Section
# ------------------------------------------------

st.markdown('<div class="section-label">Forecast Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Generate Future Projection</div>', unsafe_allow_html=True)

if st.button(f"▶  Run RNN Forecast  ·  {future_months}-Month Window"):
    with st.spinner("Analyzing temporal patterns..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)

        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    st.markdown(f"""
    <div class="forecast-banner">
        <div class="fb-icon">✅</div>
        <div class="fb-text">Forecast complete — <span>{future_months} months</span> projected from {last_date.strftime('%b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown('<div class="data-panel"><div class="data-panel-header"><span class="data-panel-title">Forecasted Values</span></div><div class="data-panel-body">', unsafe_allow_html=True)
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="↓  Export CSV",
            data=csv,
            file_name="aircast_forecast.csv",
            mime="text/csv"
        )
        st.markdown('</div></div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="data-panel"><div class="data-panel-header"><span class="data-panel-title">Combined Projection</span></div><div class="data-panel-body">', unsafe_allow_html=True)

        fig_combined = go.Figure()

        # Historical
        fig_combined.add_trace(go.Scatter(
            x=df.index,
            y=df["Passengers"],
            name="Historical",
            line=dict(color="#4A90C4", width=2),
            fill="tozeroy",
            fillcolor="rgba(74, 144, 196, 0.07)",
        ))

        # Forecast
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"],
            y=forecast_df["Predicted Passengers"],
            name="Forecast",
            line=dict(color="#F4A832", width=2.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(244, 168, 50, 0.07)",
        ))

        # Join line
        join_x = [df.index[-1], forecast_df["Month"].iloc[0]]
        join_y = [df["Passengers"].iloc[-1], forecast_df["Predicted Passengers"].iloc[0]]
        fig_combined.add_trace(go.Scatter(
            x=join_x, y=join_y,
            mode="lines",
            line=dict(color="#6B8CAE", width=1, dash="dot"),
            showlegend=False,
        ))

        fig_combined.update_layout(
            **PLOTLY_LAYOUT,
            height=330,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.12,
                xanchor="left",
                x=0,
                font=dict(size=12, color="#8BA8C4"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_combined, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)