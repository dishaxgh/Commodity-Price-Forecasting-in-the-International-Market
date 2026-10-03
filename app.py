import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMAResults

st.set_page_config(
    page_title="Commodity Price Forecasting", page_icon="📈", layout="centered"
)

st.title("Commodity Price Forecaster")
st.write(
    "Hybrid Ensemble Model (ARIMAX + LSTM Residuals) running live in the"
    " cloud."
)

COMMODITIES = {
    "Brent Oil": "Model Artifacts/Brent Oil",
    "US Soybeans": "Model Artifacts/US Soybeans",
    "US Wheat": "Model Artifacts/US Wheat",
}
LOOKBACK_WINDOW = 14


# Load models with caching so they only load once
@st.cache_resource
def load_models():
  loaded_models = {}
  for name, folder in COMMODITIES.items():
    if os.path.exists(folder):
      try:
        loaded_models[name] = {
            "arimax": ARIMAResults.load(os.path.join(folder, "arimax_model.pkl")),
            "lstm": tf.keras.models.load_model(
                os.path.join(folder, "lstm_residual_model.keras")
            ),
            "scaler": joblib.load(os.path.join(folder, "minmax_scaler.pkl")),
            "data": pd.read_csv(
                os.path.join(folder, "latest_historical_data.csv")
            ),
        }
      except Exception as e:
        st.error(f"Error loading artifacts for {name}: {e}")
  return loaded_models


models = load_models()

# Commodity selector dropdown
commodity = st.selectbox(
    "Select Commodity", list(models.keys()) if models else []
)

if st.button("Generate Next-Day Prediction"):
  if not models or commodity not in models:
    st.error("Model artifacts not found or failed to load.")
  else:
    with st.spinner("Running hybrid forecasting model..."):
      m = models[commodity]
      df = m["data"]
      arimax_model = m["arimax"]
      lstm_model = m["lstm"]
      scaler = m["scaler"]

      last_row = df.iloc[-1]
      exog_pred = pd.DataFrame(
          [[last_row["SMA_14"], last_row["Std_14"]]],
          columns=["SMA_14", "Std_14"],
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

      st.success("Prediction generated successfully!")

      col1, col2, col3 = st.columns(3)
      col1.metric("Last Date", str(last_row["Date"]))
      col2.metric("Last Price", f"${float(last_row['Price']):.2f}")
      col3.metric(
          "Predicted Next Price",
          f"${final_prediction:.2f}",
          delta=f"{round(final_prediction - float(last_row['Price']), 2)}",
      )
