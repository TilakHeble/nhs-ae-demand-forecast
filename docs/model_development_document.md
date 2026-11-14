# Model development document

## 1. Purpose and scope

| Component | Purpose | Output | Status |
|---|---|---|---|
| **Hierarchical demand forecast** | 12-month forecasts of Type 1 A&E attendances and emergency admissions via Type 1 A&E, coherent across trust → ICB → region → England | point forecasts with 80% / 95% intervals, 164 series | analytical / planning support |
| **4-hour performance risk** | probability that each trust's Type 1 4-hour performance falls below a threshold, by month | 7 quantiles per trust-month; threshold probabilities | screening / prioritisation |

Intended users are capacity and winter planners at trust, ICB, region and national level. The model is out of scope for operational decisions below monthly resolution (rotas, daily escalation), for non-Type 1 services, and for performance management of individual trusts.

## 2. Data

**Source.** NHS England, *A&E Attendances and Emergency Admissions*, monthly situation reports (Accredited Official Statistics, Open Government Licence v3.0): 103 monthly CSVs, April 2018 to October 2026. The pipeline also uses NHS England's *Provider to ICB mapping* (October 2026). The CSVs were collected through a browser, because the NHS England site isn't reachable from the build environment, and committed as one bundle with each file's URL (`data/raw/nhs_ae_monthly_bundle.txt`).

**Harmonisation** (`nhsae.ingest`):
- 18 measure columns are mapped from every historical header variant.
- TOTAL rows are dropped.
- Revised files take priority.

**Restatement on today's NHS** (`nhsae.panel`, `nhsae.reclass`):

| Issue | Count | Treatment | Check |
|---|---|---|---|
| Trust mergers | 21 retired codes | history mapped to the successor code | combined series continuous across the merger month (every pair inspected) |
| Type 1 ↔ "Other" recording changes | 32 events, 25 trusts | earlier Type 1 history rescaled by s_post / s_pre (Type 1 share of Type 1 + Other, 3 months either side) | restated daily rate continuous across the event; synthetic unit tests |
| Trust scope | 120 trusts | Type 1 attendances in the latest month and at least 36 months of history | balanced panel, no gaps |

Recording changes are detected when Type 1 changes by more than 15% between 3-month windows, "Other" moves the opposite way, and the two changes net to less than half of the Type 1 change. Windows touching January 2020 to September 2021 are skipped. 19 events moved patients out of Type 1 (median factor 0.86) and 13 moved them in.

**4-hour performance** uses the **reported** attendances and breaches, so the numerator and denominator share a recording basis. Demand forecasts use the restated series.

**Data contracts** (`nhsae.contracts`) run on every build and in CI:
- every month present, no duplicate provider-months, no negative counts
- balanced panel, breaches ≤ attendances
- each trust in exactly one ICB and each ICB in one region; 7 regions
- **no trust's daily Type 1 rate moves more than 60% month on month after July 2021**

The last check was written to catch unmapped mergers. It found the recording-change problem.

## 3. Hierarchy
y_t = S b_t, where S (164 × 120) is built from the mapping: England (1), regions (7), ICBs (36), trusts (120). Implemented in `nhsae.hierarchy.Hierarchy`.

## 4. Base models (`nhsae.models`)
Every model forecasts the **daily rate** (monthly total ÷ days in month) and converts back.

| Model | Specification |
|---|---|
| Seasonal naive | same month last year (daily rate) |
| ETS | additive damped trend + additive seasonality on log daily rate (statsmodels Holt-Winters, parameters by maximum likelihood) |
| SARIMA | ARIMA(0,1,1)(0,1,1)₁₂ on log daily rate (airline model) |
| Global LightGBM | one model across all 164 series; direct multi-horizon (h = 1…12). Target: log(rate_{o+h} / mean rate over the 12 months to o). Features: same month last year, last-3-month level, year-on-year momentum, last two months, h, target month, Easter flag, days in month, hierarchy level. 400 trees, learning rate 0.03, 31 leaves, min 40 per leaf, 0.8 row/column subsampling |
| Ensemble | equal-weight mean of ETS, SARIMA and LightGBM forecasts (and residuals) |

**COVID treatments:**
- *impute*: April 2020 to June 2021 replaced by the 2019 same-month daily rate × a ratio blending linearly from 1 to the series' July 2021–June 2022 / July 2018–June 2019 level ratio (clipped to [0.5, 2])
- *post*: train on July 2021 onwards

## 5. Reconciliation
ỹ = S G ŷ with G for bottom-up, top-down (average historical proportions), OLS, WLS (structural) and **MinT-shrink**. MinT uses W = the Schäfer–Strimmer shrinkage of the in-sample one-step residual covariance over the last 36 months. All are implemented in NumPy; coherence and the S G S = S property are tested.

## 6. Prediction intervals
- **Model-based:** Gaussian with variance diag(P W P′) × h^0.5, where P = S G and W = the shrunk residual covariance. The h^0.5 growth and the 36-month residual window were chosen from backtest coverage. Variance ∝ h gave 96–100% coverage of nominal 80% intervals.
- **Conformal option** (`nhsae.calibrate`): multiply the interval half-width by the finite-sample conformal quantile of |z| = |y − ŷ| / σ̂.
- **Selection:** four granularities (none, level × horizon band, level, pooled) are compared **online**, each calibrated at every origin only on errors already realised by then. The smallest mean |coverage − nominal| over 80% and 95% wins, per measure.
  - Attendances: model-based (gap 0.109).
  - Emergency admissions: pooled conformal (gap 0.142 → 0.057).

## 7. Model selection
**Rolling origin:** 9 origins (October 2023 to October 2025, quarterly), horizons 1–12. All 5 × 6 × 2 combinations are scored for both measures. The selection criterion is mean MASE over the four hierarchy levels, which gives each level equal weight.

| Measure | Winner | England | Region | ICB | Trust |
|---|---|---|---|---|---|
| Attendances | ensemble · MinT · post-COVID | 0.396 | 0.459 | 0.613 | 0.662 |
| Emergency admissions | ensemble · MinT · imputed | 0.377 | 0.522 | 0.752 | 0.882 |

The live forecast refits the winner on all data to October 2026.

## 8. 4-hour performance risk (`nhsae.four_hour`)
- **Target:** Type 1 share within 4 hours (including booked appointments), trust-month.
- **Model:** global LightGBM, quantile objective at τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}, direct multi-horizon. 500 trees, learning rate 0.03, 15 leaves, min 50 per leaf.
- **Features** (all known at the origin):
  - performance: last month, last 3 months, last 12 months, same month last year, year-on-year change
  - national Type 1 performance over the last 3 months
  - demand pressure: forecast target-month daily rate ÷ last-12-month mean, plus the same ratio a year earlier
  - admissions per attendance over the last 3 months
  - h, month, log trust size
- **Training:** targets and origins inside March 2020 to June 2021 are excluded. Training uses actual demand pressure; the backtest uses the reconciled demand forecast available at each origin.
- **Post-processing:**
  - quantiles sorted (no crossing)
  - conformalised quantile regression (each quantile shifted by its own residual quantile), applied online
  - CDF interpolated between quantiles with Gaussian tails, giving P(Y < threshold)

## 9. Software
- Python package `nhsae` (`pyproject.toml`).
- `make data backtest forecast calibrate four_hour notebooks`.
- CI runs ruff, rebuilds the panel from raw files through the data contracts, then runs pytest (unit, results, dashboard and course tests).
- Notebooks are built from reviewed `.py` sources (`scripts/build_notebooks.py`).
