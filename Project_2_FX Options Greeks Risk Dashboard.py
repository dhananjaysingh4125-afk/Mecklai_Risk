from scipy.stats import norm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def gk_price(S, K, T, rd, rf, sigma, option_type="call"):
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S*np.exp(-rf*T)*norm.cdf(d1)  - K*np.exp(-rd*T)*norm.cdf(d2)
    else:
        return K*np.exp(-rd*T)*norm.cdf(-d2) - S*np.exp(-rf*T)*norm.cdf(-d1)


def gk_greeks(S, K, T, rd, rf, sigma, option_type="call"):
    d1  = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2  = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    sign = 1 if option_type == "call" else -1        

    delta = sign * np.exp(-rf*T) * norm.cdf(sign*d1)
    gamma = np.exp(-rf*T) * nd1 / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-rf*T) * nd1 * np.sqrt(T) / 100
    theta = (-S*np.exp(-rf*T)*nd1*sigma/(2*np.sqrt(T))
             + sign*(rf*S*np.exp(-rf*T)*norm.cdf(sign*d1)
             - rd*K*np.exp(-rd*T)*norm.cdf(sign*d2))) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ── Portfolio ────────────────────────────────────────────────────────────────
portfolio = pd.DataFrame([
    {"pair":"USD/INR","S":83.5,"K":84.0,"T":90/365, "sigma":0.065,"type":"call","notional":1e6},
    {"pair":"EUR/INR","S":90.2,"K":89.0,"T":180/365,"sigma":0.072,"type":"put", "notional":5e5},
    {"pair":"GBP/INR","S":105.,"K":104.,"T":60/365, "sigma":0.068,"type":"call","notional":7e5},
    {"pair":"USD/INR","S":83.5,"K":82.0,"T":30/365, "sigma":0.060,"type":"put", "notional":3e5},
])

# Add tenor_bucket BEFORE pivot  ← was missing entirely
def tenor_bucket(T):
    days = T * 365
    if days <= 30:   return "0-1M"
    elif days <= 90: return "1-3M"
    elif days <= 180:return "3-6M"
    else:            return "6M+"

portfolio["tenor_bucket"] = portfolio["T"].apply(tenor_bucket)

# Compute Greeks
for i, row in portfolio.iterrows():
    g = gk_greeks(float(row.S), float(row.K),float(row["T"]),
                  0.065, 0.055, float(row.sigma), row.type)
    for k, v in g.items():
        portfolio.loc[i, k] = v * row.notional

print(portfolio[["pair","tenor_bucket","delta","gamma","vega","theta"]].round(2))


# ── Plots ────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["Delta Heatmap", "Vega by Pair", "Theta Decay", "Portfolio Delta Gauge"],
    specs=[[{"type":"heatmap"},  {"type":"bar"}],
           [{"type":"scatter"}, {"type":"indicator"}]]
)

# 1. Delta heatmap
pivot = portfolio.pivot_table(values="delta", index="pair",
                               columns="tenor_bucket", aggfunc="sum").fillna(0)
fig.add_trace(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                          y=pivot.index.tolist(), colorscale="RdBu",
                          zmid=0), row=1, col=1)

# 2. Vega bar
vega_by_pair = portfolio.groupby("pair")["vega"].sum().reset_index()
fig.add_trace(go.Bar(x=vega_by_pair["pair"], y=vega_by_pair["vega"],
                     marker_color="steelblue"), row=1, col=2)

# 3. Theta decay curve
tenors = np.linspace(1/365, 180/365, 100)
thetas = [gk_greeks(83.5, 84.0, t, 0.065, 0.055, 0.065)["theta"] * 1e6
          for t in tenors]
fig.add_trace(go.Scatter(x=tenors*365, y=thetas, mode="lines",
                          line=dict(color="tomato")), row=2, col=1)

# 4. Gauge
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=round(portfolio["delta"].sum() / 1e6, 3),
    title={"text": "Portfolio Delta (USD M)"},
    gauge={"axis":      {"range": [-5, 5]},
           "bar":       {"color": "steelblue"},
           "steps":     [{"range": [-5, -3], "color": "salmon"},
                         {"range": [3, 5],   "color": "salmon"}],
           "threshold": {"line": {"color": "red", "width": 3}, "value": 4}}
), row=2, col=2)

fig.update_layout(height=750, title_text="FX Options Greeks Dashboard",
                  showlegend=False)
fig.show()