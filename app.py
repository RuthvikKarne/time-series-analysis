import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

st.set_page_config(page_title="Airline Forecast", page_icon="✈️", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("✈️ Airline Forecast")
    future_months = st.slider("Forecast horizon (months)", 1, 24, 12)

# --- Data ---
df = DataLoader("data/airline-passengers.csv").load_data()

# --- Metrics ---
st.subheader("Model accuracy")
mae, mse, rmse = Evaluator().evaluate()
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")

st.divider()

# --- EDA ---
st.subheader("Historical data")
col_l, col_r = st.columns([1, 2])

with col_l:
    st.dataframe(df, height=320, use_container_width=True)

with col_r:
    fig = go.Figure(go.Scatter(x=df.index, y=df["Passengers"]))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320,
                      xaxis_title=None, yaxis_title="Passengers")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Forecast ---
st.subheader("Forecast")

if st.button("Run forecast"):
    with st.spinner("Running RNN model…"):
        future = Forecaster().forecast(future_months)
        future_dates = pd.date_range(
            start=df.index[-1] + pd.DateOffset(months=1),
            periods=future_months, freq="MS"
        )
        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    col_l2, col_r2 = st.columns([1, 2])

    with col_l2:
        st.dataframe(forecast_df, use_container_width=True)
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

    with col_r2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Passengers"],
                                  name="Historical"))
        fig2.add_trace(go.Scatter(x=forecast_df["Month"],
                                  y=forecast_df["Predicted Passengers"],
                                  name="Forecast"))
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360,
                           hovermode="x unified",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)