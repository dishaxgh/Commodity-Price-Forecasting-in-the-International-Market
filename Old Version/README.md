# Commodity Price Forecasting in the International Market - Research Archive 

This directory houses the foundational exploratory data analysis and individual baseline models developed during the early research phase of the project (pre-MLOps upgrade). It contains individual Jupyter notebooks testing various machine learning, deep learning, and statistical time-series models across international agricultural and energy commodities (*Brent Oil*, *US Soybeans*, and *US Wheat*).

---

## Directory Structure (`Old Version/`)

```text
Old Version/
│
├── Data/
│   ├── Brent Oil Futures Historical Data.csv
│   ├── US Soybeans Futures Historical Data.csv
│   └── US Wheat Futures Historical Data.csv
│
├── Brent Oil/                # Commodity-specific experiment notebooks
│   ├── 1. RandomForestGradientBoostDecisionTree(Chronological Split).ipynb
│   ├── 2. RandomForestGradientBoostDecisionTree(98-2).ipynb
│   ├── 3. LinearRegression(Chronological Split).ipynb
│   ├── 4. LinearRegression(98-2).ipynb
│   ├── 5. PolyRegression(Chronological Split).ipynb
│   ├── 6. PolyRegression(98-2).ipynb
│   ├── 7. SupportVectorRegressor(Chronological Split).ipynb
│   ├── 8. SupportVectorRegressor(98-2).ipynb
│   ├── 9. LSTM(Chronological Split).ipynb
│   ├── 10. LSTM(98-2).ipynb
│   ├── 11. ARIMA(Chronological Split).ipynb
│   ├── 12. ARIMA(98-2).ipynb
│   ├── 13. ARIMA(99.4-0.6).ipynb
│   └── Prophet.ipynb
│
├── US Soybeans/              # (Mirrors experiment notebooks per commodity)
└── US Wheat/                 # (Mirrors experiment notebooks per commodity)
