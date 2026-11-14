# Learn time series forecasting, from basics to hierarchical reconciliation

A self-contained course built on the data in this repository: real NHS A&E demand for England, 7 regions, 36 ICBs and 120 trusts. Every chapter explains the idea, gives the maths you need (no more), and ends with code you can run from the repo root.

| # | Chapter | You will be able to… |
|---|---|---|
| 01 | [What a time series is](01_time_series_basics.md) | decompose a series into trend, seasonality and noise; read an ACF; explain stationarity |
| 02 | [Benchmarks and honest evaluation](02_benchmarks_and_evaluation.md) | build naive benchmarks; run a rolling-origin backtest; choose between MAE, RMSE, MAPE and MASE |
| 03 | [Exponential smoothing (ETS)](03_exponential_smoothing.md) | go from simple smoothing to damped Holt-Winters; know why we model the log daily rate |
| 04 | [ARIMA and SARIMA](04_arima.md) | difference a series, read ACF/PACF, fit and diagnose a seasonal ARIMA |
| 05 | [Machine learning for time series](05_ml_for_time_series.md) | build a global LightGBM model; choose direct or recursive forecasting; avoid leakage |
| 06 | [Structural breaks: COVID and mergers](06_structural_breaks.md) | handle interventions by imputing, truncating or adding dummies; restate history after mergers |
| 07 | [Probabilistic forecasts](07_probabilistic_forecasting.md) | produce and score intervals and quantiles (CRPS, pinball, coverage); conformal calibration |
| 08 | [Hierarchical forecasting and reconciliation](08_hierarchical_reconciliation.md) | build the summing matrix; derive bottom-up, top-down, OLS, WLS and MinT; know when each wins |
| 09 | [Interview questions](09_interview_questions.md) | answer the time series questions data science interviews actually ask |

**How to use it:** read the chapters in order the first time. Run every code block. Each one is short and uses `data/processed/trust_month_panel.csv`, so you are always working with real data. Install requirements with `pip install -r requirements-dev.txt`.

**Shared helper.** Most examples start with:

```python
import pandas as pd
p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
eng = p.groupby("month")["attendances"].sum().asfreq("MS")   # England Type 1 attendances
```

**Further reading:** Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., free online at otexts.com/fpp3) is the best single book. Chapters 3, 5, 8, 9 and 11 map onto chapters 01, 02, 03, 04 and 08 here.
