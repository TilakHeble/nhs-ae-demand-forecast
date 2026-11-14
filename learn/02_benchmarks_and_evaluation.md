# 02 · Benchmarks and honest evaluation

Most forecasting mistakes in industry are **evaluation mistakes**, not modelling mistakes. A model that "gets 2% MAPE" means nothing without two things: (a) a benchmark to compare against, and (b) a test that respects time.

## 1. Benchmarks every model must beat

| Method | Forecast for every future month | When it's hard to beat |
|---|---|---|
| Mean | average of all history | stationary noise |
| Naive | last observed value | random walks (stock prices) |
| **Seasonal naive** | same month last year | strongly seasonal, stable series, like A&E |
| Drift | last value + average historical change | steady trend |

In this project **seasonal naive** is the benchmark. If a sophisticated model can't beat "same month last year", it isn't worth deploying.

## 2. Never use a random train/test split

Random K-fold cross-validation leaks the future into the past: the model trains on March 2025 and is tested on February 2025. Use one of these instead:

- **Holdout**: train on everything before a cut-off, test after it. Simple, but one cut-off is one sample and can be lucky.
- **Rolling origin** (also called time series cross-validation or "evaluation on a rolling forecasting origin"): pick several cut-offs ("origins"), and at each one train only on data up to that point and forecast h steps ahead. Average the errors.

```
origin 1: [train..........] → forecast 12 months
origin 2: [train.............] → forecast 12 months
origin 3: [train................] → forecast 12 months
```

`src/nhsae/backtest.py` uses 9 origins, every 3 months from October 2023 to October 2025, each forecasting 12 months ahead.

## 3. Error metrics

With error e = y − ŷ:

| Metric | Formula | Pros | Cons |
|---|---|---|---|
| MAE | mean \|e\| | interpretable, robust | scale-dependent, so you can't compare a trust with England |
| RMSE | √mean e² | penalises big misses | scale-dependent; dominated by outliers |
| MAPE | mean \|e/y\| | unit-free, intuitive | explodes near zero; penalises over- and under-forecasts asymmetrically |
| **MASE** | MAE / MAE of in-sample seasonal naive | unit-free, defined at zero, has a natural benchmark (MASE < 1 means you beat seasonal naive) | less intuitive to explain to non-technical audiences |

MASE (Hyndman & Koehler, 2006) is the standard metric for comparing across series of different sizes. That is exactly the situation in a hierarchy where England is about 300 times bigger than a small trust.

## Try it: a mini rolling-origin backtest

```python
import pandas as pd, numpy as np
p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
eng = p.groupby("month")["attendances"].sum().asfreq("MS")

def snaive(train, h):
    rate = train / train.index.days_in_month
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    return pd.Series(np.resize(rate.iloc[-12:].to_numpy(), h) * idx.days_in_month, idx)

def drift(train, h):
    rate = train / train.index.days_in_month
    slope = (rate.iloc[-1] - rate.iloc[-13]) / 12          # last year's change per month
    base = snaive(train, h) / snaive(train, h).index.days_in_month
    idx = base.index
    return (base + slope * 12) * idx.days_in_month          # same month last year + one year of growth

rows = []
for origin in pd.date_range("2023-10-01", "2025-10-01", freq="3MS"):
    train, test = eng.loc[:origin], eng.loc[origin + pd.offsets.MonthBegin():].iloc[:12]
    scale = np.mean(np.abs(train.iloc[-24:].to_numpy() - train.iloc[-36:-12].to_numpy()))
    for name, f in {"seasonal naive": snaive, "snaive + drift": drift}.items():
        e = test - f(train, 12).loc[test.index]
        rows.append({"method": name, "MAPE": np.mean(np.abs(e / test)), "MASE": np.mean(np.abs(e)) / scale})
print(pd.DataFrame(rows).groupby("method").mean().round(3))
```

Notice that the "obvious improvement" is **worse**: seasonal naive scores MASE 0.70 and the drift version 0.84. Last month's year-on-year change is a noisy estimate of growth, and extrapolating it adds error. That is exactly why every idea, even a common-sense one, goes through the backtest. The models in `src/` estimate the trend from many months at once, and they beat both.

## 4. Choosing the winner fairly

- Choose the model **only** from the backtest, never from the final test period.
- Compare at the level the decision is made. Trust-level accuracy matters for bed plans; England-level accuracy matters for national funding.
- Report the benchmark next to every number.
- If you try 30 configurations, the best one is partly lucky. A clear and consistent gap across levels and origins is more convincing than one big win.
