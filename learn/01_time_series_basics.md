# 01 · What a time series is

A **time series** is a sequence of observations ordered in time: y₁, y₂, …, y_T. What makes it different from ordinary tabular data is that **the order carries information**. Last month tells you a lot about this month. Shuffling the rows, which is harmless in a cross-sectional dataset, destroys the signal.

Our main series is monthly Type 1 A&E attendances in England, from April 2018 to October 2026.

## 1. The three components

Almost every series can be described as some mix of:

| Component | What it is | In A&E data |
|---|---|---|
| **Trend** | long-run movement in the level | population growth and ageing push demand up about 2–4% a year |
| **Seasonality** | a pattern that repeats with a fixed period | summer peaks, February troughs (fewer days), winter acuity |
| **Remainder (noise)** | what's left | heatwaves, strikes, IT outages, reporting quirks |

They combine either **additively**, y = T + S + R, where seasonal swings stay the same absolute size, or **multiplicatively**, y = T × S × R, where swings grow with the level. Taking logs turns multiplicative into additive: log y = log T + log S + log R. That is one reason so many models in this repo work on the log scale.

### Try it: STL decomposition

STL ("Seasonal and Trend decomposition using Loess") separates the components robustly.

```python
import pandas as pd, numpy as np
from statsmodels.tsa.seasonal import STL

p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
eng = p.groupby("month")["attendances"].sum().asfreq("MS")
daily = eng / eng.index.days_in_month          # remove the days-in-month effect (see §3)
post = np.log(daily.loc["2021-07":])           # post-COVID period, log scale

res = STL(post, period=12, robust=True).fit()
strength_trend = max(0, 1 - res.resid.var() / (res.trend + res.resid).var())
strength_season = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
print(f"trend strength {strength_trend:.2f}, seasonal strength {strength_season:.2f}")
print(res.seasonal.groupby(res.seasonal.index.month).mean().round(3))  # seasonal effect by calendar month (log points)
```

The **strength** measures (FPP3 §4.3) go from 0 to 1. Values near 1 mean that component dominates. The seasonal effect by month shows which months run above or below the trend: 0.03 is roughly 3% above.

## 2. Autocorrelation

**Autocorrelation at lag k** is the correlation between y_t and y_{t−k}. Plotting it for k = 1, 2, … gives the **ACF**.

- A slowly decaying ACF means a **trend** (non-stationary).
- Spikes at lags 12, 24, … in a monthly series mean **seasonality**.
- An ACF with nothing outside the ±1.96/√T bands means **white noise**: nothing left to forecast.

```python
from statsmodels.tsa.stattools import acf
print(pd.Series(acf(post, nlags=24), name="acf").round(2).iloc[[1, 2, 3, 6, 12, 24]])
```

## 3. Calendar effects: forecast the daily rate

February has 28 days and March has 31, so February "drops" by about 10% even if nothing changed. If you model raw monthly totals, the model wastes effort learning this. **Divide by days in the month, forecast the daily rate, then multiply back.** Every model in `src/models.py` does this. It is a two-line fix that removes a whole source of error.

Easter is the other big calendar effect for health data. It moves between March and April and changes how services run around bank holidays. The LightGBM model in this repo gets an "Easter in this month" feature for that reason.

## 4. Stationarity

A series is (weakly) **stationary** if its mean, variance and autocorrelation don't change over time. Most statistical theory, ARIMA especially, assumes stationarity. Real demand series aren't stationary, so we transform them:

- **log** to stabilise variance
- **first difference**, y_t − y_{t−1}, to remove trend
- **seasonal difference**, y_t − y_{t−12}, to remove seasonality

The **KPSS** test has "stationary" as its null hypothesis, and the **ADF** test has "unit root (non-stationary)" as its null. Using both is good practice.

```python
from statsmodels.tsa.stattools import kpss, adfuller
import warnings; warnings.simplefilter("ignore")
for name, s in {"log level": post, "seasonal diff": post.diff(12).dropna(),
                "seasonal + first diff": post.diff(12).diff().dropna()}.items():
    print(f"{name:22s} KPSS p={kpss(s, nlags='auto')[1]:.3f}   ADF p={adfuller(s)[1]:.3f}")
```

A small KPSS p-value (< 0.05) rejects stationarity. A small ADF p-value rejects non-stationarity. You want the series where KPSS is large and ADF is small.

## Key takeaways

- Look before you model. Plot the series, decompose it and check the ACF.
- Logs turn multiplicative structure into additive structure.
- Remove known calendar effects (days in month, Easter) rather than asking a model to learn them.
- Differencing is how ARIMA handles trend and seasonality (chapter 04). Exponential smoothing models them directly instead (chapter 03).
