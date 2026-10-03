from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import pandas as pd
import tensorflow as tf

app = FastAPI(title="Commodity Price Forecasting API", version="1.0")

COMMODITIES = {
    "Brent Oil": "Brent Oil",
    "US Soybeans": "US Soybeans",
    "US Wheat": "US Wheat",
}
LOOKBACK_WINDOW = 14

# Load models into memory on startup
models = {}
for name, folder in COMMODITIES.items():
  if os.path.exists(folder):
    try:
      models[name] = {
          "arimax": joblib.load(os.path.join(folder, "arimax_model.pkl")),
          "lstm": tf.keras.models.load_model(
              os.path.join(folder, "lstm_residual_model.keras")
          ),
          "scaler": joblib.load(os.path.join(folder, "minmax_scaler.pkl")),
          "data": pd.read_csv(os.path.join(folder, "latest_historical_data.csv")),
      }
    except Exception as e:
      print(f"Error loading artifacts for {name}: {e}")


@app.get("/")
def home():
  return {
      "message": "Commodity Price Forecasting API is running.",
      "available_commodities": list(models.keys()),
  }


@app.post("/predict/{commodity_name}")
def predict_next_price(commodity_name: str):
  if commodity_name not in models:
    raise HTTPException(
        status_code=404, detail=f"Commodity '{commodity_name}' not found."
    )

  m = models[commodity_name]
  df = m["data"]
  arimax_model = m["arimax"]
  lstm_model = m["lstm"]
  scaler = m["scaler"]

  if len(df) < LOOKBACK_WINDOW:
    raise HTTPException(
        status_code=400, detail="Insufficient historical data for inference."
    )

  last_row = df.iloc[-1]
  exog_pred = pd.DataFrame(
      [[last_row["SMA_14"], last_row["Std_14"]]], columns=["SMA_14", "Std_14"]
  )

  # 1. ARIMAX prediction
  arimax_pred = arimax_model.forecast(steps=1, exog=exog_pred).iloc[0]

  # 2. LSTM residual prediction
  recent_prices = df["Price"].values[-LOOKBACK_WINDOW:]
  fitted_vals = arimax_model.fittedvalues.tail(LOOKBACK_WINDOW).values
  recent_residuals = (recent_prices - fitted_vals).reshape(-1, 1)

  scaled_res = scaler.transform(recent_residuals)
  X_input = np.reshape(scaled_res, (1, LOOKBACK_WINDOW, 1))

  scaled_pred_res = lstm_model.predict(X_input, verbose=0)
  lstm_pred = scaler.inverse_transform(scaled_pred_res)[0][0]

  # 3. Combine hybrid prediction
  final_prediction = float(arimax_pred + lstm_pred)

  return {
      "commodity": commodity_name,
      "last_date": str(last_row["Date"]),
      "last_price": float(last_row["Price"]),
      "predicted_next_price": round(final_prediction, 2),
  }
