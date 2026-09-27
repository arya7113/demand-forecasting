"""Streamlit dashboard: 28-day demand forecast vs actual sales for Walmart store CA_1."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("28-Day Demand Forecasting — Walmart Store CA_1")
st.caption("XGBoost model trained on M5 sales data. The last 28 days were hidden from training and are shown here as the test period.")

forecasts = pd.read_parquet(ART / "test_forecasts.parquet")
history = pd.read_parquet(ART / "history.parquet")
metrics = json.loads((ART / "metrics.json").read_text())

st.subheader("Accuracy on the unseen 28 days (WAPE, lower is better)")
table = pd.DataFrame(metrics).T[["dept_wape", "item_wape"]]
table.columns = ["Department level", "Item level"]
st.dataframe(table.style.format("{:.1%}"))

level = st.radio("View", ["Department", "Single item"], horizontal=True)
if level == "Department":
    choice = st.selectbox("Department", sorted(forecasts.dept_id.unique()))
    f = forecasts[forecasts.dept_id == choice]
    h = history[history.dept_id == choice]
else:
    choice = st.selectbox("Item", sorted(forecasts.item_id.unique()))
    f = forecasts[forecasts.item_id == choice]
    h = history[history.item_id == choice]

past = h.groupby("date")["sales"].sum().rename("Actual sales")
future = f.groupby("date")[["sales", "forecast", "moving_average_28d"]].sum()
future.columns = ["Actual sales", "XGBoost forecast", "28-day average"]
chart = pd.concat([past.to_frame(), future])
st.line_chart(chart)

total_actual, total_fc = future["Actual sales"].sum(), future["XGBoost forecast"].sum()
c1, c2 = st.columns(2)
c1.metric("Actual units (28 days)", f"{total_actual:,.0f}")
c2.metric("Forecast units (28 days)", f"{total_fc:,.0f}", f"{(total_fc / total_actual - 1):+.1%} vs actual")
