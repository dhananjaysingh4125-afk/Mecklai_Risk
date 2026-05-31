from scipy.stats import norm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

scenarios = {
  "INR Crisis (2013-style Taper)": {
      "usdinr_chg": +0.18,  # INR weakens 18%
      "fed_rate_chg": +0.01, # +100bps
      "oil_chg": +0.15,      # oil up 15%
      "nifty_chg": -0.12     # equity down 12%
  },
  "Global Recession (2008-style)": {
      "usdinr_chg": +0.22,
      "fed_rate_chg": -0.03,
      "oil_chg": -0.55,
      "nifty_chg": -0.55
  },
  "INR Appreciation (IT Boom)": {
      "usdinr_chg": -0.10,
      "fed_rate_chg": -0.005,
      "oil_chg": -0.10,
      "nifty_chg": +0.20
  },
  "Oil Shock (2022 Russia)": {
      "usdinr_chg": +0.05,
      "fed_rate_chg": +0.02,
      "oil_chg": +0.70,
      "nifty_chg": -0.08
  },
  "Soft Landing (Goldilocks)": {
      "usdinr_chg": -0.03,
      "fed_rate_chg": -0.01,
      "oil_chg": -0.05,
      "nifty_chg": +0.12
  },
}

spot        = 83.50        # Current spot
notional    = 2_000_000   # USD
fwd_rate    = 83.80        # Locked forward rate
hedge_ratio = 0.60        # 60% hedged

for sc_name, sc in scenarios.items():
    new_spot   = spot * (1 + sc["usdinr_chg"])
    
    # Unhedged portion P&L
    unhedged   = notional * (1 - hedge_ratio) * (new_spot - spot)
    
    # Hedged portion: locked at fwd_rate, gain/loss vs new spot
    hedged     = notional * hedge_ratio * (fwd_rate - new_spot)
    
    net_pnl    = unhedged + hedged
    print(f"{sc_name}: Net P&L = INR {net_pnl:,.0f}")

loss_threshold = -20_000_000   # INR -2 crore

for depreciation in np.arange(0.01, 0.30, 0.005):
    new_s   = spot * (1 + depreciation)
    pnl     = (notional*(1-hedge_ratio)*(new_s-spot) +
               notional*hedge_ratio*(fwd_rate-new_s))
    if pnl < loss_threshold:
        print(f"Breaking point: INR depreciates {depreciation:.1%}")
        print(f"New spot: {new_s:.2f} | Loss: INR {pnl:,.0f}")
        break

