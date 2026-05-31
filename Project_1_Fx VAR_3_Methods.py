import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, chi2

# ── 1. Data ──────────────────────────────────────────────────────────────────
prices  = yf.download("USDINR=X", start="2020-01-01", end="2025-01-01")["Close"].squeeze()
log_ret = np.log(prices / prices.shift(1)).dropna().squeeze()

# ── 2. $50M Importer Book ─────────────────────────────────────────────────────
position    = -50_000_000   # Short USD (importer)
port_std_1d = log_ret.std()
scaling     = np.sqrt(10)   # 1-day → 10-day scaling

# ── 3. METHOD 1 — Parametric (Delta-Normal) ───────────────────────────────────
VaR_param_1d  = abs(position) * norm.ppf(0.99) * port_std_1d
VaR_param_10d = VaR_param_1d * scaling

print("=" * 50)
print("METHOD 1 — PARAMETRIC")
print(f"  1-day  VaR (99%): USD {VaR_param_1d:>12,.0f}")
print(f"  10-day VaR (99%): USD {VaR_param_10d:>12,.0f}")

# ── 4. METHOD 2 — Historical Simulation ──────────────────────────────────────
pnl_hist      = log_ret * position          # P&L for each historical day
VaR_hs_1d     = -np.percentile(pnl_hist, 1)
VaR_hs_10d    = VaR_hs_1d * scaling

# Expected Shortfall (CVaR) — beyond VaR tail
ES_99         = -pnl_hist[pnl_hist < -VaR_hs_1d].mean()

print("\nMETHOD 2 — HISTORICAL SIMULATION")
print(f"  1-day  VaR (99%): USD {VaR_hs_1d:>12,.0f}")
print(f"  10-day VaR (99%): USD {VaR_hs_10d:>12,.0f}")
print(f"  Expected Shortfall: USD {ES_99:>10,.0f}")

# ── 5. METHOD 3 — Monte Carlo ─────────────────────────────────────────────────
np.random.seed(42)
n_sims        = 100_000
mu            = log_ret.mean()
sigma         = log_ret.std()

# Simulate 10-day cumulative return paths
sim_10d_ret   = np.random.normal(mu, sigma, (n_sims, 10)).sum(axis=1)
sim_pnl_10d   = sim_10d_ret * position

VaR_mc_10d    = -np.percentile(sim_pnl_10d, 1)
ES_mc_99      = -sim_pnl_10d[sim_pnl_10d < -VaR_mc_10d].mean()

print("\nMETHOD 3 — MONTE CARLO (100,000 simulations)")
print(f"  10-day VaR (99%): USD {VaR_mc_10d:>12,.0f}")
print(f"  Expected Shortfall: USD {ES_mc_99:>10,.0f}")

# ── 6. Iran-Oil Stress ────────────────────────────────────────────────────────
usdinr_shock  = 0.08                          # INR weakens 8%
stressed_pnl  = position * usdinr_shock       # importer loss
stressed_spot = 83.5 * (1 + usdinr_shock)

print("\nIRAN-OIL STRESS (Brent +40%, DXY +8%)")
print(f"  Stressed Spot:       {stressed_spot:.2f}")
print(f"  Stress Loss:     USD {abs(stressed_pnl):>12,.0f}")
print(f"  Exceeds MC VaR by:   USD {abs(stressed_pnl) - VaR_mc_10d:>10,.0f}")

# ── 7. Summary Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"{'Method':<25} {'10d VaR (99%)'}")
print("-" * 50)
print(f"{'Parametric':<25} USD {VaR_param_10d:>12,.0f}")
print(f"{'Historical Simulation':<25} USD {VaR_hs_10d:>12,.0f}")
print(f"{'Monte Carlo':<25} USD {VaR_mc_10d:>12,.0f}")
print(f"{'Iran-Oil Stress Loss':<25} USD {abs(stressed_pnl):>12,.0f}")

# ── 8. Kupiec POF Backtest (on Historical method) ────────────────────────────
def kupiec_pof(n=250, confidence=0.99):
    # Count actual exceptions over last 250 days
    rolling_var  = pnl_hist.rolling(250).quantile(0.01) * -1
    actual_pnl   = pnl_hist.iloc[-250:]
    var_forecast = rolling_var.iloc[-250:]
    exceptions   = int((actual_pnl < -var_forecast).sum())

    p  = 1 - confidence
    if exceptions == 0:
        LR = 0.0
    else:
        LR = 2 * (
            np.log((exceptions/n)**exceptions * (1 - exceptions/n)**(n-exceptions))
          - np.log(p**exceptions * (1-p)**(n-exceptions))
        )

    critical = chi2.ppf(0.95, df=1)   # 3.841
    passed   = LR < critical
    zone     = ("GREEN ✅"  if exceptions <= 4
                else "YELLOW ⚠️" if exceptions <= 9
                else "RED ❌")

    print("\nKUPIEC POF BACKTEST (250 days)")
    print(f"  Exceptions:      {exceptions} / 250")
    print(f"  LR Statistic:    {LR:.4f}  |  Critical: {critical:.4f}")
    print(f"  Kupiec POF:      {'PASS ✅' if passed else 'FAIL ❌'}")
    print(f"  Basel Zone:      {zone}")

kupiec_pof()