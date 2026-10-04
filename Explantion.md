## Overview of the Hybrid Ensemble Framework

Financial commodity prices (such as Brent Oil, US Soybeans, and US Wheat) are notoriously difficult to forecast because they exhibit a combination of smooth, 
linear macroeconomic trends & sharp, non-linear shocks caused by sudden market panic or supply chain disruptions. 
Traditional linear models fail to capture complex volatility, while pure machine learning models often overfit noisy time-series data. The exact reason why we had to
upgrade the previous research; mainly because the models used could not extrapolate beyond the dataset distribution, for example, tree-based models.

To solve this, the project implements a Hybrid Ensemble Framework that divides forecasting into two stages:   

**1. Linear Trend Modeling (ARIMAX):** Captures the primary autoregressive trend and integrates external market indicators.
**2. Non-Linear Residual Correction (LSTM):** Learns from the residual errors left behind by the linear model to capture high-frequency fluctuations.

### 1. Component 1: ARIMAX Model 
The ARIMAX (AutoRegressive Integrated Moving Average with Exogenous Inputs) model serves as the primary linear baseline. 
In the project code, it is configured with an order of $(5, 1, 0)$ using statsmodels, 

where, **AR ($p=5$):** Uses the previous 5 days of price history to predict the current price.

**I ($d=1$):** Applies first-order differencing ($P_t - P_{t-1}$) to make non-stationary price series stationary (removing unit roots/trends).

**MA ($q=0$):** Moving average lag component is set to zero in this configuration.  

#### Mathematical Assumptions & Role
1. Linearity Assumption: ARIMAX assumes that future values are a linear combination of past values, past errors, and external drivers.
2. Stationarity: Assumes that after differencing ($d=1$), the time series has a constant mean and variance over time.
3. Role: It establishes a stable baseline by capturing the macro-level directional movement of the commodity price.

### 2. Exogenous Features: SMA_14 and Std_14

To enhance the ARIMAX baseline, two rolling window features are supplied as exogenous variables (exog): 
##### A. 14-Day Simple Moving Average (SMA₁₄)

**Formula:**

$$
\mathrm{SMA}_{14,t} = \frac{1}{14}\sum_{i=0}^{13} P_{t-i}
$$

where:

- $P_t$ is the closing price on day $t$.
- $P_{t-i}$ represents the closing price $i$ days before day $t$.
- $\mathrm{SMA}_{14,t}$ is the 14-day simple moving average at day $t$.

**Role:** It smooths out short-term daily noise and random price spikes, helping the ARIMAX model capture the underlying 2-week medium-term directional trend.

##### B. 14-Day Rolling Standard Deviation (Std₁₄)

**Formula:**

$$
\mathrm{Std}_{14,t} = \sqrt{\frac{1}{13}\sum_{i=0}^{13}\left(P_{t-i} - \mathrm{SMA}_{14,t}\right)^2}
$$

where:

- $P_t$ is the closing price on day $t$.
- $P_{t-i}$ represents the closing price $i$ days before day $t$.
- $\mathrm{SMA}_{14,t}$ is the 14-day simple moving average at day $t$.
- $\mathrm{Std}_{14,t}$ is the rolling 14-day standard deviation at day $t$.
- The denominator $13$ corresponds to the **sample standard deviation** with $14$ observations.

**Role:** It acts as a market volatility gauge. A high standard deviation indicates greater price variability, uncertainty, or potential market shocks, enabling the ARIMAX model to adjust its predictions during volatile periods.

### 3. Component 2: LSTM Neural Network 

Once ARIMAX fits the data, it leaves behind prediction errors known as residuals:

$$
\mathrm{Residuals}_t = \mathrm{Price}_t - \mathrm{ARIMAX\_Fitted}_t
$$

These residuals represent everything the linear model failed to capture (complex non-linear patterns, sudden shocks, and high-frequency noise). 
An LSTM neural network is trained exclusively on these residuals. 

##### Architecture & Mechanics

**Input Window:** Uses a lookback window of 14 days (LOOKBACK_WINDOW = 14) to sequence historical error patterns.   
**Layers:** Built with two LSTM layers (50 units each) featuring a 0.2 dropout rate for regularization to prevent overfitting, followed by a Dense output layer.   
**Role:** LSTMs are specialized Recurrent Neural Networks (RNNs) designed to retain long-term memory via gating mechanisms, allowing them to detect hidden temporal dependencies and non-linear patterns in market error sequences. 

### 4. Hybrid Combination & Final Prediction
The final output of the framework is an additive combination of both models:

$$
\mathrm{FinalPrediction}_t = \mathrm{ARIMAXForecast}_t - \mathrm{LSTMResidualForecast}_t
$$

By combining linear baseline forecasting with deep learning residual correction, the framework achieves superior forecasting accuracy across volatile commodities 
like Brent Oil, US Soybeans, and US Wheat.


## Understanding Walk-Forward Time-Series Cross-Validation (`TimeSeriesSplit`)

In financial time-series forecasting, standard cross-validation techniques such as random **K-Fold Cross-Validation** are inappropriate because they randomly shuffle observations. This can allow the model to train on future data points while predicting past observations—a form of **data leakage** that produces falsely optimistic backtest results and may fail when deployed in live markets.

To prevent this, the project implements **Walk-Forward Time-Series Cross-Validation** using `scikit-learn`'s `TimeSeriesSplit(n_splits=N_SPLITS)`.

This ensures that the model is always trained strictly on **past observations** and evaluated exclusively on **unseen future observations**.

### How `TimeSeriesSplit` Prevents Data Leakage

The `run_walk_forward_cv` function systematically prevents data leakage across every layer of the hybrid forecasting pipeline.

### 1. Strict Chronological Slicing

**What happens:**
```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1): splits the dataset sequentially.
```

**Prevention mechanism:** The training set (train_df) always consists of historical observations that occurred before the test set (test_idx). The model is never allowed to look ahead into future market behavior.

### 2. Out-of-Sample ARIMAX Forecasting

**What happens:** The ARIMAX model is fitted solely on the training partition: 
```python
arimax_fit = ARIMA(train_df['Price'], exog=exog_train, order=(5,1,0)).fit().   
```
**Prevention mechanism:** Predictions for the test period are generated out-of-sample using arimax_fit.predict(start=test_idx[0], end=test_idx[-1], exog=exog_test). The exogenous features (SMA_14 and Std_14) for the test set are isolated and only supplied to forecast future steps, meaning the ARIMAX coefficients never see the test target values during training. 

### 3. Isolated MinMax Scaling of Residuals

**What happens:** The code calculates training residuals 
```python
(train_res = train_df['Price'] - train_df['ARIMAX_Fitted']).
```
A MinMaxScaler is then instantiated and fitted strictly on the training residuals: 
```python
scaler.fit_transform(train_res).   
```
**Prevention mechanism:** When scaling the test residuals, the scaler is not re-fitted. Instead, it uses .transform(combined_res) based on the historical training 
bounds. This ensures that the minimum and maximum values of the future test set do not leak into the scaling parameters. 

### 4. Controlled LSTM Sequence Construction (create_sequences)
**What happens:** The LSTM requires a rolling lookback window **(LOOKBACK_WINDOW = 14)** to form input sequences. For the test set,
the code bridges the gap using past training context: 
```python
combined_res = np.vstack((train_res[-lookback:], true_test_res)).
```
**Prevention mechanism:** By appending the last 14 residuals of the training set to the front of the test residuals, 
the LSTM can evaluate the first test day without crossing the boundary line or leaking future test targets into past sequences.

### 5. No Validation Split Leakage During Training
**What happens:** The LSTM is trained via 
```python
lstm_cv.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
```
with no validation split parameter (validation_data or validation_split).   

**Prevention mechanism:** In standard deep learning, validation splits are often used for early stopping, which can indirectly leak information about the 
validation distribution into the training loop. By training blindly for a fixed 20 epochs on past data alone, the model remains entirely isolated from 
future evaluation metrics.

Basically, by enforcing chronological boundaries through TimeSeriesSplit, isolating feature scalers to training folds, and generating strict out-of-sample 
forecasts for both ARIMAX and LSTM components, the framework guarantees that its performance metrics (RMSE and MAE) reflect true, real-world predictive 
capability rather than data leakage artifacts.














