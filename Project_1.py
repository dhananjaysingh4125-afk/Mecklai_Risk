
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Currency pairs: USDINR, EURINR
pairs = ["USDINR=X", "EURINR=X"]
prices_Open = yf.download(pairs, start="2025-02-01", end="2025-05-20")["Open"]
prices_High = yf.download(pairs, start="2025-02-01", end="2025-05-20")["High"]
prices_Low = yf.download(pairs, start="2025-02-01", end="2025-05-20")["Low"]
prices_Close = yf.download(pairs, start="2025-02-01", end="2025-05-20")["Close"]
prices = (prices_Open + prices_High + prices_Low + prices_Close)/4 # Using avg prices for VaR calculations (Volatility)   
# Log returns for each pair
log_ret = np.log(prices / prices.shift(1)).dropna()
eur_usd = 1.08
# Notional positions in USD (positive = long, negative = short)
positions = {
    "USDINR=X": 1000000,   # Long USD 1M (Exporter hedge)
    "EURINR=X": -250000/eur_usd,   # Short EUR 250K (Importer hedge)
}
pos_vec = np.array([positions[p] for p in pairs])

from scipy.stats import norm

cov_matrix = log_ret.cov()
weights    = pos_vec / pos_vec.sum()

port_variance = weights @ cov_matrix @ weights
port_std      = np.sqrt(port_variance)

VaR_param_95 = pos_vec.sum() * norm.ppf(0.95) * port_std
VaR_param_99 = pos_vec.sum() * norm.ppf(0.99) * port_std
print(f"Parametric VaR (99%): USD {VaR_param_99:,.0f}")
plt.hist(log_ret, bins=10)
# 3. Add labels and a title
plt.title('Return Histogram (Normal Distribution)')
plt.xlabel('Values')
plt.ylabel('Frequency')
# 4. Display the plot
plt.show()

# Portfolio P&L for each historical day
pnl_hist = (log_ret * pos_vec).sum(axis=1)

VaR_hs_95 = -np.percentile(pnl_hist, 5)
VaR_hs_99 = -np.percentile(pnl_hist, 1)

# Also compute Expected Shortfall (CVaR)
ES_99 = -pnl_hist[pnl_hist < -VaR_hs_99].mean()
print(f"Historical VaR (99%): USD {VaR_hs_99:,.0f}")
print(f"Expected Shortfall:   USD {ES_99:,.0f}")

n_sims   = 100_000
mean_ret = log_ret.mean().values
cov_np   = log_ret.cov().values

# Simulate correlated returns
sim_returns = np.random.multivariate_normal(mean_ret, cov_np, n_sims)

sim_pnl     = sim_returns @ pos_vec
VaR_mc_99   = -np.percentile(sim_pnl, 1)
print(f"Monte Carlo VaR (99%): USD {VaR_mc_99:,.0f}")

# Rolling 1-year HS VaR backtest
window     = 252
exceptions = 0

for i in range(window, len(pnl_hist)):
    hist_window = pnl_hist.iloc[i-window:i]
    var_today   = -np.percentile(hist_window, 1)
    actual_pnl  = pnl_hist.iloc[i]
    if actual_pnl < -var_today:
        exceptions += 1

print(f"VaR Exceptions (last year): {exceptions}")
print("Basel Zone: ", "GREEN" if exceptions < 5 else "YELLOW")
