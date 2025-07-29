import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # ✅ Filter only data from 2023
    df = df[df["Year"] == 2023]

    # Encode Weather Condition as numeric codes if needed
    if "Weather Condition" in df.columns:
        df["Weather Condition"] = df["Weather Condition"].astype("category").cat.codes

    return df

def train_rf(df):
    df = df.copy()
    df = df.sort_values("Date")

    features = ["Year", "Month", "Day", "Weather Condition", "Holiday/Promotion", "Price"]

    for f in features:
        if f not in df.columns:
            raise ValueError(f"Missing feature in dataset: {f}")

    X = df[features]
    y = df["Inventory Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("✅ Train RMSE:", train_rmse)
    print("✅ Test RMSE:", test_rmse)
    print("✅ Overfitting Detected:", test_rmse > train_rmse * 1.5)

    return model

def forecast_inventory(df, model, days, future_weather, future_promo, future_price):
    df = df.copy()
    df = df.sort_values("Date")
    last_date = df["Date"].max()

    weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Snowy": 3}
    weather_code = weather_map.get(future_weather, 0)
    promo_code = future_promo

    forecasts = []
    for i in range(1, days + 1):
        date = last_date + pd.Timedelta(days=i)
        features = [date.year, date.month, date.day, weather_code, promo_code, future_price]
        X_pred = np.array(features).reshape(1, -1)
        pred = model.predict(X_pred)[0]
        forecasts.append([date, pred])

    future_df = pd.DataFrame(forecasts, columns=["Date", "Forecast"])
    actual_df = df[["Date", "Inventory Level"]].copy()
    actual_df["Forecast"] = np.nan
    combined = pd.concat([actual_df, future_df], ignore_index=True)
    return combined

def check_inventory_alerts(forecast_df, threshold):
    future = forecast_df[forecast_df["Inventory Level"].isna()]
    alerts = future[future["Forecast"] < threshold]
    return alerts

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
