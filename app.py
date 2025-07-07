# app.py

import streamlit as st
from utils import (
    load_data,
    preprocess_data,
    train_rf,
    forecast_inventory,
    check_inventory_alerts,
    save_model,
    get_feature_importance,
    get_test_predictions
)
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Inventory Forecasting", layout="wide")
st.title("📈 Retail Store Inventory Forecasting — Random Forest")

# Load & preprocess
data = load_data("data/retail_store_inventory.csv")
data = preprocess_data(data)

# Sidebar
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
threshold = st.sidebar.number_input("Inventory Alert Threshold", min_value=1, value=50)

store_ids = sorted(data["Store ID"].unique())
categories = sorted(data["Category"].unique())

store_id = st.sidebar.selectbox("Store ID", store_ids)
category = st.sidebar.selectbox("Category", categories)

df = data[(data["Store ID"] == store_id) & (data["Category"] == category)]

if df.empty:
    st.warning("No data for this Store ID + Category.")
    st.stop()

if st.checkbox("Show raw data"):
    st.dataframe(df)

st.subheader(f"🔍 Forecast: Store ID {store_id} | Category {category}")

# Train + forecast
model, rmse = train_rf(df)
save_model(model, f"models/rf_{store_id}_{category}.pkl")

forecast_df = forecast_inventory(df, model, forecast_days)
alerts = check_inventory_alerts(forecast_df, threshold)

# 📈 Forecast chart
st.line_chart(forecast_df.set_index("Date")[["Inventory Level", "Forecast"]])

# Show only future: Date + Forecast
future_forecast = forecast_df[forecast_df["Inventory Level"].isna()][["Date", "Forecast"]]
st.subheader("🔮 Future Forecast")
st.dataframe(future_forecast)

# ⚠️ Alerts
if not alerts.empty:
    st.warning(f"⚠️ {len(alerts)} alert(s)! Forecast below {threshold}")
    st.dataframe(alerts[["Date", "Forecast"]])

# ✅ Feature Importance with friendly names
st.subheader("🧩 Feature Importance")

importances = get_feature_importance(model)

# Technical names:
features = ["Lag1", "Rolling7", "Year", "Month", "Day"]
# Friendly display names:
friendly_names = ["Previous Day Inventory", "7-Day Average Inventory", "Year", "Month", "Day of Month"]

importance_df = pd.DataFrame({
    "Feature": friendly_names,
    "Importance": importances
})
st.bar_chart(importance_df.set_index("Feature"))

## ✅ Actual vs. Predicted for test
st.subheader("✅ Actual vs. Predicted (Last Test Period)")

y_test, preds = get_test_predictions(df, model)
fig, ax = plt.subplots(figsize=(8, 3))  # <-- ADJUST SIZE HERE
ax.plot(y_test.values, label="Actual", marker="o")
ax.plot(preds, label="Predicted", marker="x")
ax.set_title("Last 7 Days - Actual vs. Predicted")
ax.legend()
st.pyplot(fig)
# ✅ Show RMSE
st.subheader("✅ Model Performance")
st.write(f"**Root Mean Squared Error (RMSE)**: `{rmse:.2f}`")


# Download only future forecast
csv = future_forecast.to_csv(index=False).encode()
st.download_button("Download Future Forecast", csv, "future_forecast.csv", "text/csv")
