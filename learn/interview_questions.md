# Interview questions on time series

Short model answers, each tied to something in this repo you can point to.

**1. Why can't you use random K-fold cross-validation on a time series?**
It trains on the future to predict the past, so errors look better than they will be in production. Use rolling-origin evaluation: train up to a cut-off, forecast forward, and repeat for several cut-offs. *Repo: `src/nhsae/backtest.py`, 9 origins × 12 horizons.*

**2. What's a good benchmark for monthly demand?**
Seasonal naive (same month last year), or seasonal naive with drift. Report MASE, which is the error relative to seasonal naive, so it's clear whether the model earns its complexity.

**3. MAPE or MASE?**
MAPE is intuitive but breaks near zero and penalises over-forecasts more than under-forecasts. MASE is scale-free, defined at zero, and has a built-in benchmark. For comparing a trust with England, MASE.

**4. Explain ETS in one breath.**
Weighted averages of the past with exponentially decaying weights, maintained separately for level, trend and seasonality. Trend damping stops long-range forecasts running away. It's a state-space model, so it has a likelihood and intervals.

**5. How do you choose ARIMA orders?**
Make the series stationary (log, seasonal and first differences; check with KPSS and ADF), then read the ACF and PACF, or search with AICc. Always check that the residuals are white noise (Ljung–Box).

**6. Recursive or direct multi-step forecasting?**
Recursive feeds predictions back in as lags, so errors compound. Direct predicts y_{t+h} from information at t, with h as a feature or one model per horizon. Direct avoids compounding and is simpler to keep leak-free.

**7. What is a global model and why use one?**
One model trained across many series using scale-free features. It learns shared patterns from far more data than any single series has. It needs normalisation by each series' own level.

**8. How did you handle COVID?**
I compared truncating to post-July 2021 against imputing a counterfactual, and let the backtest choose for each measure. Truncation won for attendances, imputation for emergency admissions. The point is that it's an empirical choice.

**9. What is forecast reconciliation?**
Base forecasts at every level don't add up. Reconciliation maps them to coherent forecasts, ỹ = S G ŷ. Bottom-up, top-down, OLS, WLS and MinT differ only in G. MinT uses the error covariance and minimises total error variance.

**10. Does reconciliation hurt accuracy?**
Usually the opposite: levels borrow strength from each other. In this repo MinT beat the unreconciled forecasts at England level for both measures. Top-down was the method that hurt.

**11. How do you know your prediction intervals are any good?**
Check coverage against nominal and use a proper score such as CRPS or pinball loss. If coverage is off, calibrate, for example with split conformal on past forecast errors, and evaluate that online so you aren't calibrating on the test set.

**12. What's the difference between a prediction interval and a confidence interval?**
A confidence interval covers a parameter, such as the mean. A prediction interval covers a future observation, so it also includes the noise and is wider.

**13. How would you forecast 4-hour performance, which is a ratio?**
Ratios don't add up a hierarchy, so I didn't reconcile it. I used a quantile LightGBM with recent performance, national pressure and the demand forecast as features. Then I turned the quantiles into P(performance < threshold) and checked calibration with Brier scores. *Repo: `src/nhsae/four_hour.py`.*

**14. A stakeholder says "the forecast was wrong last month". How do you respond?**
Ask whether the outcome was inside the interval. A point forecast is the middle of a distribution, and 1 month in 5 should fall outside an 80% interval. If misses are systematic (bias), investigate. Emergency admissions here have run below forecast since spring 2024, which is worth raising with operational teams rather than tuning away.

**15. What would you do with more time or data?**
Daily ECDS data for staffing-level forecasts; weather and flu surveillance as leading indicators; forecasting site-level Type 3 activity; scenario forecasts for winter planning.
