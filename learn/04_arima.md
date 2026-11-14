# 04 · ARIMA and SARIMA

Where exponential smoothing describes trend and seasonality directly, **ARIMA describes the autocorrelation** that remains after differencing has made the series stationary.

## 1. The pieces

- **AR(p), autoregressive:** y_t depends on its own past values.
  y_t = c + φ₁y_{t−1} + … + φ_p y_{t−p} + ε_t
- **I(d), integrated:** difference d times to remove trend.
- **MA(q), moving average:** y_t depends on past *forecast errors*.
  y_t = c + ε_t + θ₁ε_{t−1} + … + θ_q ε_{t−q}

**ARIMA(p,d,q)** combines them. **Seasonal ARIMA**, ARIMA(p,d,q)(P,D,Q)_m, adds the same structure at the seasonal lag m (12 for monthly data).

## 2. Identifying orders from ACF and PACF

After differencing to stationarity:

| Pattern | ACF | PACF |
|---|---|---|
| AR(p) | decays gradually | cuts off after lag p |
| MA(q) | cuts off after lag q | decays gradually |
| Seasonal MA(1) | single spike at lag 12 | spikes at 12, 24, … decaying |

In practice you often use automatic selection (`auto_arima`, or AICc search), but being able to read these plots is an interview staple.

## 3. The airline model

**ARIMA(0,1,1)(0,1,1)₁₂** on the log series is the "airline model" (Box & Jenkins, 1970, fitted to airline passenger data). It is equivalent to a particular exponential smoothing model and is a famously strong default for monthly seasonal data. That's what `src/nhsae/models.py` uses: one robust specification applied to all 164 series, rather than 164 separately tuned models that would overfit.

## Try it: fit and diagnose

```python
import pandas as pd, numpy as np, warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import pacf, acf
warnings.simplefilter("ignore")

p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
eng = p.groupby("month")["attendances"].sum().asfreq("MS")
z = np.log(eng / eng.index.days_in_month).loc["2021-07":"2025-10"]

d = z.diff(12).diff().dropna()
print("ACF  lags 1,12:", acf(d, nlags=12)[[1, 12]].round(2), " PACF lags 1,12:", pacf(d, nlags=12)[[1, 12]].round(2))

fit = SARIMAX(z, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print(fit.summary().tables[1])
lb = acorr_ljungbox(fit.resid.iloc[13:], lags=[12], return_df=True)
print("Ljung-Box p-value (residuals white noise if > 0.05):", round(float(lb.lb_pvalue.iloc[0]), 3))
```

**Residual diagnostics** are the step people skip. If the Ljung–Box test rejects white noise, the model has left structure behind and its intervals will be wrong.

## 4. ETS or ARIMA?

- Both handle trend and seasonality, and neither dominates in general.
- ETS has no stationarity requirement and is easier to explain. ARIMA can include external regressors (that makes it SARIMAX) and models short-term autocorrelation more flexibly.
- **Combining them usually beats choosing between them.** This project's best attendance forecasts average ETS, SARIMA and LightGBM.

**COVID gives a lesson here.** The airline model differences twice, so a big shock 12 months ago echoes into this year's forecast through the seasonal difference. That is why SARIMA benefited most from replacing the COVID months with a counterfactual. Its MASE improved from 0.634 to 0.563 when trained on imputed history rather than truncated data.
