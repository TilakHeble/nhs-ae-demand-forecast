# 03 · Exponential smoothing (ETS)

Exponential smoothing forecasts are **weighted averages of past observations, with weights that decay exponentially into the past**. Recent data counts most. It's 70 years old, fast, robust, and still wins forecasting competitions as part of ensembles.

## 1. The ladder

**Simple exponential smoothing (SES): level only**

    ℓ_t = α y_t + (1 − α) ℓ_{t−1}          forecast: ŷ_{t+h} = ℓ_t

α ∈ (0, 1) controls memory. α close to 1 means "mostly last value"; close to 0 means "long average".

**Holt: level + trend**

    ℓ_t = α y_t + (1 − α)(ℓ_{t−1} + b_{t−1})
    b_t = β (ℓ_t − ℓ_{t−1}) + (1 − β) b_{t−1}       forecast: ℓ_t + h b_t

**Damped trend.** Multiply the trend by φ^h (0.8 < φ < 1) so the forecast flattens out over long horizons. This is one of the most reliable ideas in forecasting: trends rarely continue forever, and a damped trend prevents runaway long-range forecasts.

**Holt-Winters: + seasonality**

    ŷ_{t+h} = ℓ_t + (φ + φ² + … + φ^h) b_t + s_{t+h−m}

with a seasonal component s updated like the level, and m = 12 for monthly data.

## 2. The ETS framework

Hyndman et al. write every variant as **ETS(Error, Trend, Seasonal)**:

- **E**rror: additive (A) or multiplicative (M)
- **T**rend: none (N), additive (A) or additive damped (Ad)
- **S**easonal: none (N), additive (A) or multiplicative (M)

So damped Holt-Winters with multiplicative seasonality is ETS(M,Ad,M). Because these are state-space models with a likelihood, you can compare variants with AIC and get prediction intervals.

**This repo uses ETS(A,Ad,A) on the log daily rate.** Additive seasonality on the log scale is multiplicative seasonality on the original scale, and the log makes the errors roughly constant in size.

## Try it

```python
import pandas as pd, numpy as np, warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.simplefilter("ignore")

p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
eng = p.groupby("month")["attendances"].sum().asfreq("MS")
train, test = eng.loc[:"2025-10"], eng.loc["2025-11":"2026-10"]
z = np.log(train / train.index.days_in_month)

for damped in [False, True]:
    fit = ExponentialSmoothing(z, trend="add", damped_trend=damped, seasonal="add",
                               seasonal_periods=12, initialization_method="estimated").fit()
    fc = np.exp(fit.forecast(12)) * test.index.days_in_month
    mape = np.mean(np.abs(test.to_numpy() - fc.to_numpy()) / test.to_numpy())
    print(f"damped={damped!s:5}  alpha={fit.params['smoothing_level']:.2f} "
          f"phi={fit.params.get('damping_trend', 1):.2f}  12-month MAPE={mape:.2%}  AIC={fit.aic:.1f}")
```

Read the fitted parameters. A high α means the level reacts quickly to new data. φ shows how strongly the trend is damped. On this single origin the undamped model happens to win. One origin is one sample, though (chapter 02). Damping is the widely recommended default for longer horizons (FPP3 §8.2) because it limits the damage when a trend stops, and that is why the repo uses it. A good exercise is to add an undamped ETS to `src/nhsae/models.py` and see what the full backtest says.

## 3. Practical notes

- **Transform first** (log, daily rate). ETS on raw monthly totals has to learn February from scratch.
- **Seasonal periods need at least 2 full cycles** of data. We have 103 months, which is plenty.
- **COVID breaks the level.** Exponential smoothing adapts quickly (high α), which helps after a shock but also means a recent anomaly can pull forecasts off. Chapter 06 covers the fixes.
- In this project ETS on its own was beaten by the ensemble, but it carried a lot of the ensemble's accuracy. It is a dependable component.
