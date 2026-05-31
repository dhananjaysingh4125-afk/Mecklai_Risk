from scipy.stats import norm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

pairs = ["USDINR=X", "EURINR=X", "USDJPY=X"]
prices = yf.download(pairs, start="2022-01-01", end="2025-05-28")["Close"]
log_ret = np.log(prices / prices.shift(1)).dropna()
# Realized Volatility (annualized)
rv_5d  = log_ret["USDINR=X"].rolling(5).std() * np.sqrt(252)
rv_22d = log_ret["USDINR=X"].rolling(22).std() * np.sqrt(252)

# Features: lagged RV, lagged returns, day-of-week, VIX proxy
df = pd.DataFrame({
    "rv_5d":    rv_5d,
    "rv_22d":   rv_22d,
    "ret_1d":   log_ret["USDINR=X"],
    "rv_lag1":  rv_5d.shift(1),
    "rv_lag5":  rv_5d.shift(5),
}).dropna()


from arch import arch_model

returns    = log_ret["USDINR=X"] * 100   # Scale for numerical stability
garch_mdl = arch_model(returns, vol="GARCH", p=1, q=1, dist="t")
res       = garch_mdl.fit(disp="off")

# 5-day vol forecast
forecast  = res.forecast(horizon=5)
vol_5d    = np.sqrt(forecast.variance.iloc[-1].sum()) / 100 * np.sqrt(252)
print(f"GARCH 5d Annualized Vol Forecast: {vol_5d:.2%}")


egarch = arch_model(returns, vol="EGARCH", p=1, q=1, o=1, dist="t")
eres   = egarch.fit(disp="off")

# Gamma parameter: if negative, downside moves spike vol more
gamma  = eres.params["gamma[1]"]
print(f"Leverage Effect (gamma): {gamma:.4f}")
print("INR depreciation amplifies vol:", gamma < 0)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build sequence inputs (lookback window = 20 days)
def create_sequences(data, target, lookback=20):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(20, 3)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── 1. Data ──────────────────────────────────────────────────────────────────
prices  = yf.download("USDINR=X", start="2020-01-01", end="2025-01-01")["Close"].squeeze()
log_ret = np.log(prices / prices.shift(1)).dropna().squeeze()

# Verify both are Series before building df
print(type(prices))   # should be <class 'pandas.core.series.Series'>
print(type(log_ret))  # should be <class 'pandas.core.series.Series'>

# ── 2. Realized Volatility (target variable) ─────────────────────────────────
rv_5d  = log_ret.rolling(5).std()  * np.sqrt(252)
rv_22d = log_ret.rolling(22).std() * np.sqrt(252)

df = pd.DataFrame({
    "rv_5d":   rv_5d,
    "rv_22d":  rv_22d,
    "ret_1d":  log_ret,
    "rv_lag1": rv_5d.shift(1),
    "rv_lag5": rv_5d.shift(5),
}).dropna()

# ── 3. GARCH(1,1) ─────────────────────────────────────────────────────────────
returns   = log_ret * 100
garch_mdl = arch_model(returns, vol="GARCH", p=1, q=1, dist="t")
garch_res = garch_mdl.fit(disp="off")

egarch_mdl = arch_model(returns, vol="EGARCH", p=1, q=1, o=1, dist="t")
egarch_res = egarch_mdl.fit(disp="off")

# ── 4. Walk-forward split (80/20, no random shuffle) ─────────────────────────
split      = int(len(df) * 0.80)
train_df   = df.iloc[:split]
test_df    = df.iloc[split:]

# GARCH & EGARCH predictions on test set
garch_preds  = garch_res.conditional_volatility.iloc[split:].values  / 100 * np.sqrt(252)
egarch_preds = egarch_res.conditional_volatility.iloc[split:].values / 100 * np.sqrt(252)
y_test_garch = df["rv_5d"].iloc[split:].values   # same target for all models

# ── 5. LSTM — sequence builder ────────────────────────────────────────────────
LOOKBACK = 20
FEATURES = ["rv_5d", "rv_lag1", "rv_lag5"]   # 3 input features
TARGET   = "rv_5d"

# Scale
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_all = scaler_X.fit_transform(df[FEATURES].values)
y_all = scaler_y.fit_transform(df[[TARGET]].values)

def create_sequences(X, y, lookback=20):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_all, y_all, LOOKBACK)

# Walk-forward split on sequences
split_seq  = int(len(X_seq) * 0.80)
X_train    = X_seq[:split_seq]
y_train    = y_seq[:split_seq]
X_test     = X_seq[split_seq:]
y_test_seq = y_seq[split_seq:]

# ── 6. Build & Train LSTM ─────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# ── 7. LSTM predictions — inverse scale back to actual vol ───────────────────
lstm_preds_scaled = model.predict(X_test)
lstm_preds        = scaler_y.inverse_transform(lstm_preds_scaled).flatten()
y_test_lstm       = scaler_y.inverse_transform(y_test_seq).flatten()

# ── 8. Align all three models to same test length ────────────────────────────
min_len      = min(len(garch_preds), len(egarch_preds), len(lstm_preds))
garch_preds  = garch_preds[-min_len:]
egarch_preds = egarch_preds[-min_len:]
lstm_preds   = lstm_preds[-min_len:]
y_test       = y_test_lstm[-min_len:]     # ← single aligned y_test for all

# ── 9. Model Comparison ───────────────────────────────────────────────────────
print(f"\n{'Model':<10} {'RMSE':>8} {'MAE':>8} {'DirAcc':>8}")
print("-" * 38)
for name, preds in {"GARCH": garch_preds, "EGARCH": egarch_preds, "LSTM": lstm_preds}.items():
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    da   = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y_test)))
    print(f"{name:<10} {rmse:>8.4f} {mae:>8.4f} {da:>8.1%}")

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

from sklearn.metrics import mean_squared_error, mean_absolute_error

for name, preds in {"GARCH": garch_preds, 
                    "EGARCH": egarch_preds, 
                    "LSTM": lstm_preds}.items():
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    
    # Directional accuracy
    da   = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y_test)))
    print(f"{name}: RMSE={rmse:.4f} | MAE={mae:.4f} | DirAcc={da:.1%}")