# Commodity Price Forecasting in the International Market

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://commodity-price-forecasting-in-the-international-market-ijltpt.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end MLOps pipeline and hybrid ensemble architecture designed to forecast international market commodity prices (*Brent Oil*, *US Soybeans*, and *US Wheat*) by combining econometric linear modeling and deep learning residual forecasting.

**Live Web Application:** [Access Streamlit Cloud App](https://commodity-price-forecasting-in-the-international-market-ijltpt.streamlit.app/)

---

## Project Upgrade: from Previous IISER Project

This repository represents a major architectural, methodological, and production-level upgrade over the previous **IISER project**. While the initial IISER project established a foundational exploratory data analysis and basic predictive baseline, this version transforms the work into a robust, deployable MLOps application.

### What Changed and Why?

| Component | Previous IISER Project | Upgraded MLOps Pipeline | Why the Change Was Made |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Standard standalone models (e.g., basic ARIMA & LSTM). | **Hybrid Ensemble (ARIMAX + LSTM Residuals)**. | Commodities exhibit both long-term linear economic trends and chaotic, non-linear shocks. ARIMAX captures linear baseline movements with exogenous technical indicators (`SMA_14`, `Std_14`), while the LSTM targets remaining errors (residuals) to maximize precision. |
| **Validation Strategy** | Single static train-test split or arbitrary evaluation. | **Walk-Forward Time-Series Cross-Validation (`TimeSeriesSplit`)**. | Financial and time-series data must respect chronological order. Walk-forward cross-validation prevents data leakage and rigorously evaluates performance across multiple rolling folds. |
| **Model Persistence** | Trained live in notebooks with no saved production artifacts. | **Automated Serialization (`.pkl` & `.keras`)**. | Enables true production readiness by decoupling model training (`train.py`) from cloud inference (`app.py`), preventing the need to retrain models from scratch on every user request. |
| **User Interface & Deployment** | Static Jupyter notebook outputs and local plots. | **Cloud-Native Interactive Web App (Streamlit Community Cloud)**. | Elevates the project from a static research report into an accessible, interactive production tool for live forecasting. |

---

## Architectural Overview

Financial commodity prices exhibit both linear trends (handled well by traditional statistical methods) and volatile non-linear patterns/shocks. To maximize forecasting accuracy, this project implements a **Hybrid Ensemble Framework**:

1. **ARIMAX (Linear Baseline):** Captures the primary autoregressive trend and integrates external exogenous features (`SMA_14` - 14-day Simple Moving Average, and `Std_14` - 14-day Rolling Standard Deviation).
2. **LSTM Neural Network (Non-linear Residual Forecaster):** Fits on the residuals (errors) extracted from the ARIMAX model to catch remaining non-linear temporal dynamics and high-frequency fluctuations.
3. **Hybrid Combination:** Final prediction = $\text{ARIMAX Forecast} + \text{LSTM Residual Forecast}$.

---

## Repository Structure

```text
Commodity-Price-Forecasting-in-the-International-Market/
│
├── Model Artifacts/
│   ├── Brent Oil/
│   │   ├── arimax_model.pkl
│   │   ├── lstm_residual_model.keras
│   │   ├── minmax_scaler.pkl
│   │   └── latest_historical_data.csv
│   ├── US Soybeans/
│   │   ├── arimax_model.pkl
│   │   ├── lstm_residual_model.keras
│   │   ├── minmax_scaler.pkl
│   │   └── latest_historical_data.csv
│   └── US Wheat/
│       ├── arimax_model.pkl
│       ├── lstm_residual_model.keras
│       ├── minmax_scaler.pkl
│       └── latest_historical_data.csv
│
├── Data/                        # Raw historical data CSVs
├── train.py                     # Monolithic training & pipeline generation script
├── app.py                       # Unified Streamlit Cloud deployment script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
