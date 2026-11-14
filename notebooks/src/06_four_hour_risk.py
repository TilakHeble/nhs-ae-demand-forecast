# %% [markdown]
# # 06 · From demand to waits: 4-hour performance risk by trust
#
# **Question:** which trusts are most likely to see Type 1 4-hour performance fall below a threshold this winter,
# and can we trust those probabilities?
#
# 4-hour performance is a ratio, so it can't be reconciled up the hierarchy like demand. `nhsae.four_hour` fits a
# global LightGBM at 7 quantiles (5th–95th percentile), direct multi-horizon, with conformalised quantiles.
# Demand pressure enters as the reconciled demand forecast for the target month relative to the trust's recent level.
# In the backtest that is the forecast available at each origin, never the actual.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nhsae import viz
from nhsae.config import PANEL, RESULTS

viz.set_style()
summ = pd.read_csv(RESULTS / "four_hour_backtest_summary.csv", index_col=0)
summ.round(4)

# %%
mae = summ["MAE (pct points)"].dropna()
brier = summ.loc[summ.index.str.startswith("Brier")]
skill = 1 - brier["Brier"] / brier["Brier (climatology)"]
print(f"median forecast error {mae.iloc[0]:.2f} pp vs best naive {mae.iloc[1:].min():.2f} pp")
print("Brier skill vs base rate:", skill.map("{:.0%}".format).to_dict())

# %% [markdown]
# ## 1. Calibration of the event probabilities

# %%
cal = pd.read_csv(RESULTS / "four_hour_calibration.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, ev in zip(axes, cal.event.unique()):
    c = cal[cal.event == ev]
    ax.plot([0, 1], [0, 1], ls=":", color=viz.NHS_GREY)
    ax.plot(c.predicted, c.observed, marker="o", color=viz.NHS_BLUE)
    for _, r in c.iterrows():
        ax.annotate(f"n={r.n}", (r.predicted, r.observed), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set_title(f"P({ev})", fontsize=10)
    ax.set_xlabel("predicted probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("observed frequency")
fig.suptitle("Calibration of 4-hour risk probabilities (backtest)", fontweight="bold")
viz.save(fig, "06_calibration")

# %% [markdown]
# ## 2. What drives the forecast

# %%
imp = pd.read_csv(RESULTS / "four_hour_feature_importance.csv", index_col=0)["gain_share"]
imp.round(3)

# %% [markdown]
# A trust's own recent performance dominates: flow problems (beds, discharge, staffing) persist from month to month.
# National pressure is next. The demand forecast adds little at monthly resolution. That fits notebook 02:
# A&E footfall is not what peaks in winter.

# %% [markdown]
# ## 3. This winter

# %%
lv = pd.read_csv(RESULTS / "four_hour_forecast.csv", parse_dates=["target_month"])
w = lv[lv.target_month.dt.month.isin([12, 1, 2])]
w = w[w.target_month <= w.target_month.min() + pd.DateOffset(months=2)]
risk = (w.groupby(["trust_code", "trust_name", "region_name"])
        .agg(median=("q50", "mean"), p_below_60=("p_below_60", "mean"), p_worse=("p_worse_than_ly", "mean"))
        .reset_index())
print(f"Winter {w.target_month.min():%b %Y}–{w.target_month.max():%b %Y}: median trust forecast {risk['median'].median():.1%}; "
      f"{(risk.p_below_60 > 0.5).sum()} of {len(risk)} trusts more likely than not below 60%; "
      f"{(risk.p_worse > 0.5).sum()} more likely than not worse than last winter")
risk.sort_values("p_worse", ascending=False).head(10).round(3)

# %%
perf = pd.read_csv(PANEL, parse_dates=["month"])
reg_last = (perf[perf.month.dt.month.isin([12, 1, 2]) & (perf.month >= "2025-12-01") & (perf.month <= "2026-02-01")]
            .groupby("region_name")[["over_4h", "attendances_reported"]].sum())
reg_last = 1 - reg_last.over_4h / reg_last.attendances_reported
reg_now = risk.groupby("region_name")["median"].median()
order = reg_now.sort_values().index
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, r in enumerate(order):
    pts = risk[risk.region_name == r]["median"]
    ax.scatter(np.full(len(pts), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(pts)), 100 * pts,
               color=viz.NHS_BLUE, alpha=0.6, s=18)
    ax.scatter(i, 100 * reg_last[r], marker="_", s=400, color=viz.NHS_RED, lw=2)
ax.axhline(60, color=viz.NHS_GREY, ls=":", lw=1)
ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
ax.set_ylabel("Type 1 % within 4 hours")
ax.set_title("Forecast winter Type 1 4-hour performance by trust (dots); red = region's actual last winter")
viz.save(fig, "06_winter_risk")
