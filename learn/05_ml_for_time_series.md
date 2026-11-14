# 05 · Machine learning for time series

Gradient boosting (LightGBM, XGBoost) won the M5 forecasting competition, but only when it was set up correctly. A time series has to be turned into a supervised learning table, and that step is where most mistakes happen.

## 1. Local or global models

- **Local:** one model per series (164 ETS fits). Each model sees only its own history.
- **Global:** **one model trained on all series at once.** The model learns shared patterns, such as "December is busy everywhere" or "trusts growing fast keep growing", from 164 × 100 observations instead of 100.

Global models need **scale-free features and targets**. A small trust and England must look alike to the model. This repo divides everything by each series' own last-12-month average and works in logs, so the target becomes "log of demand relative to its recent level".

## 2. Multi-step forecasting: recursive or direct

To forecast 12 months ahead:

- **Recursive:** train a one-step model; predict t+1, feed that prediction back in as a lag, predict t+2, and so on. Errors compound.
- **Direct:** train with horizon h as a feature (or one model per h). The target is y_{t+h}, and features use only information at time t. No error feedback.

`src/nhsae/models.py::lgbm_forecast` is **direct**. Each training row is (series, origin o, horizon h), with the target at o + h and every feature built from data up to o.

## 3. Feature engineering, the safe way

For target month o + h, from origin o:

| Feature | Why |
|---|---|
| same month last year, y_{o+h−12} | captures seasonality (available because h ≤ 12) |
| mean of last 3 months | current level |
| last 3 months vs same 3 months a year earlier | year-on-year momentum |
| horizon h, target month, Easter flag, days in month | tells the model what it is predicting |
| hierarchy level | England behaves differently from a single trust |

## 4. The leakage traps

1. **Lags shorter than the horizon.** Using y_{t−1} to predict 6 months ahead means using a value you won't have. Every feature must be computable at the origin.
2. **Scaling with future data.** Normalising with the mean of the whole series leaks the future. Use statistics up to the origin only.
3. **Random CV.** See chapter 02. Use rolling-origin validation.
4. **Target-derived features.** For example, a rolling mean that includes the target month.

## Try it: a tiny direct global model

```python
import pandas as pd, numpy as np, lightgbm as lgb

p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
y = p.pivot(index="month", columns="trust_code", values="attendances").asfreq("MS").loc["2021-07":]
rate = (y.T / y.index.days_in_month.to_numpy()).T.to_numpy()
months = y.index

def rows(o, h):
    scale = rate[o - 11:o + 1].mean(0)
    X = np.column_stack([np.log(rate[o + h - 12] / scale), np.log(rate[o - 2:o + 1].mean(0) / scale),
                         np.full(rate.shape[1], h), np.full(rate.shape[1], (months[o] + pd.DateOffset(months=h)).month)])
    return X, scale

origin = months.get_loc(pd.Timestamp("2025-10-01"))
X, Y = [], []
for o in range(14, origin):
    for h in range(1, 13):
        if o + h > origin: break
        f, s = rows(o, h); X.append(f); Y.append(np.log(rate[o + h] / s))
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1).fit(np.vstack(X), np.concatenate(Y))

errs = []
for h in range(1, 13):
    f, s = rows(origin, h)
    pred = np.exp(model.predict(f)) * s
    errs.append(np.mean(np.abs(rate[origin + h] - pred) / rate[origin + h]))
print(f"trust-level MAPE, 12 months from Oct 2025: {np.mean(errs):.1%}")
```

## 5. What the backtest says

On its own, the global LightGBM was mid-table. Statistical models are hard to beat on 100 monthly points per series. But its errors were **different**, so adding it to the ensemble improved accuracy. That is the realistic role of ML in a lot of business forecasting: a diverse ensemble member, not a magic replacement. ML pulls further ahead when you have many series, high-frequency data (daily or hourly) and rich external drivers such as weather and events.
