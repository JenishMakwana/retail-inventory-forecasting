# utils.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    if "Inventry Level" in df.columns:
        df.rename(columns={"Inventry Level": "Inventory Level"}, inplace=True)
    return df

def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    return df

def train_rf(df):
    df = df.copy()
    df["Lag1"] = df["Inventory Level"].shift(1)
    df["Rolling7"] = df["Inventory Level"].rolling(7, min_periods=1).mean()
    df = df.dropna()

    X = df[["Lag1", "Rolling7", "Year", "Month", "Day"]]
    y = df["Inventory Level"]

    X_train, X_test = X.iloc[:-7], X.iloc[-7:]
    y_train, y_test = y.iloc[:-7], y.iloc[-7:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, rmse

def forecast_inventory(df, model, days):
    df = df.copy()
    future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=days)

    last_value = df["Inventory Level"].iloc[-1]
    rolling = df["Inventory Level"].tail(7).mean()

    forecasts = []
    for date in future_dates:
        features = np.array([[last_value, rolling, date.year, date.month, date.day]])
        pred = model.predict(features)[0]
        forecasts.append([date, pred])
        last_value = pred
        rolling = (rolling * 6 + pred) / 7

    # Historical actuals
    actual_df = df[["Date", "Inventory Level"]].copy()
    actual_df["Forecast"] = np.nan

    forecast_df = pd.DataFrame(forecasts, columns=["Date", "Forecast"])
    forecast_df["Inventory Level"] = np.nan

    combined = pd.concat([actual_df, forecast_df], ignore_index=True)
    return combined

def check_inventory_alerts(forecast_df, threshold):
    future = forecast_df[forecast_df["Inventory Level"].isna()]
    alerts = future[future["Forecast"] < threshold]
    return alerts

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

# utils.py (add at the bottom)

def get_feature_importance(model):
    return model.feature_importances_

def get_test_predictions(df, model):
    df = df.copy()
    df["Lag1"] = df["Inventory Level"].shift(1)
    df["Rolling7"] = df["Inventory Level"].rolling(7, min_periods=1).mean()
    df = df.dropna()

    X = df[["Lag1", "Rolling7", "Year", "Month", "Day"]]
    y = df["Inventory Level"]

    X_test = X.iloc[-7:]
    y_test = y.iloc[-7:]
    preds = model.predict(X_test)

    return y_test, preds
