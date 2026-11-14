# Time series forecasting: cheat sheet

## Workflow
1. **Plot it.** Level, trend, seasonality, breaks, outliers.
2. **Fix the data before modelling.** Calendar effects (daily rate), organisational changes (mergers, recording changes), interventions (COVID).
3. **Transform.** Log for multiplicative structure; difference only for ARIMA.
4. **Benchmarks first.** Seasonal naive; naive; drift.
5. **Candidate models.** ETS, (S)ARIMA, a global ML model, an ensemble.
6. **Rolling-origin backtest** at the horizons and levels that decisions use.
7. **Probabilistic check.** Coverage, CRPS or pinball, then calibrate.
8. **Reconcile** if there is a hierarchy.
9. **Refit on all data, forecast, monitor bias.**

## Models in one line each
| Model | Idea | Strength | Weakness |
|---|---|---|---|
| Seasonal naive | ŷ_{t+h} = y_{t+h−m} | unbeatable baseline for stable seasonal data | no trend |
| SES / Holt / Holt-Winters | exponentially weighted level (+ trend + season) | fast, robust, interpretable | one series at a time; no regressors |
| Damped trend | trend × φ^h | safe long horizons | can under-forecast strong growth |
| ARIMA(p,d,q)(P,D,Q)_m | autocorrelation of the differenced series | flexible short-term dynamics; regressors (SARIMAX) | needs stationarity; shocks echo through seasonal differences |
| Airline model | ARIMA(0,1,1)(0,1,1)₁₂ on logs | strong default for monthly data | same as ARIMA |
| Global GBM (LightGBM) | one tabular model across all series | many series, rich features, nonlinearity | leakage risk; weak on short single series |
| Ensemble | average of diverse models | usually the most accurate | harder to explain |

## Metrics
| Metric | Formula | Use |
|---|---|---|
| MAE | mean \|e\| | one series, interpretable |
| RMSE | √mean e² | penalise big misses |
| MAPE | mean \|e/y\| | stakeholder-friendly; avoid near zero |
| **MASE** | MAE / in-sample seasonal-naive MAE | compare across scales; < 1 beats seasonal naive |
| Coverage | share of actuals inside the interval | ≈ nominal |
| Pinball | a(y−q)⁺ + (1−a)(q−y)⁺ | score a quantile |
| CRPS | ∫ (F(x) − 1{y ≤ x})² dx | score a whole distribution (= MAE for a point) |
| Brier | mean (p − outcome)² | score event probabilities; compare with climatology |

## Reconciliation
ỹ = S G ŷ   ·   S = summing matrix   ·   any G gives coherent forecasts

| Method | G |
|---|---|
| Bottom-up | [0 \| I] |
| Top-down | [p \| 0] |
| OLS | (S′S)⁻¹S′ |
| WLS | (S′Λ⁻¹S)⁻¹S′Λ⁻¹ |
| MinT | (S′W⁻¹S)⁻¹S′W⁻¹, with W = shrunk error covariance |

Unbiased methods satisfy S G S = S, so forecasts that already add up don't change.

## Traps
- Random K-fold on time series → leakage.
- Lags shorter than the horizon (direct models) → leakage.
- Normalising with full-series statistics → leakage.
- Picking the model on the final test period → optimistic results.
- Monthly totals without a days-in-month adjustment → a fake February effect.
- Mergers and recording changes read as demand shocks.
- Trusting nominal intervals without checking coverage.
- Tuning away bias that is a real-world signal.
