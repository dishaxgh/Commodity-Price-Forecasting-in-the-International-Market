import requests
import streamlit as st

st.set_page_config(
    page_title="Commodity Price Forecasting", page_icon="📈", layout="centered"
)

st.title("🌾 Commodity Price Forecaster")
st.write(
    "Hybrid Ensemble Model (ARIMAX + LSTM Residuals) serving real-time"
    " predictions."
)

# Commodity selector
commodity = st.selectbox(
    "Select Commodity", ["Brent Oil", "US Soybeans", "US Wheat"]
)

if st.button("Generate Next-Day Prediction"):
  with st.spinner("Fetching latest data and running hybrid model..."):
    try:
      response = requests.post(f"http://127.0.0.1:8000/predict/{commodity}")

      if response.status_code == 200:
        result = response.json()
        st.success("Prediction generated successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Date", result["last_date"])
        col2.metric("Last Price", f"${result['last_price']:.2f}")
        col3.metric(
            "Predicted Next Price",
            f"${result['predicted_next_price']:.2f}",
            delta=f"{round(result['predicted_next_price'] - result['last_price'], 2)}",
        )
      else:
        st.error(
            f"API Error: {response.json().get('detail', 'Unknown error')}"
        )
    except requests.exceptions.ConnectionError:
      st.error(
          "Could not connect to the FastAPI backend. Make sure `app.py` is"
          " running (`uvicorn app:app --reload`)."
      )
