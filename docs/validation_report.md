# Validation report

Independent-style review of the models in the [model development document](model_development_document.md). All results are out-of-sample (rolling origin) unless stated.

## 1. Summary

| Area | Result | Rating |
|---|---|---|
| Data integrity | contracts pass; every month present; restatement changes the England total by at most 1.2% in any month | 🟢 |
| Organisational restatement | 21 mergers and 32 recording changes, each continuous across its event | 🟢 (one site move unsplittable) |
| Coherence | trusts → ICBs → regions → England exact to 1e-8 in every live forecast | 🟢 |
| Attendance accuracy | beats seasonal naive at every level (MASE 0.40–0.66) | 🟢 |
| Emergency-admission accuracy | beats seasonal naive at every level (MASE 0.38–0.88); persistent over-forecast since April 2024 | 🟡 bias flagged |
| Attendance intervals | 80% coverage 77–88% across levels; slightly wide at England level | 🟢 |
| Emergency-admission intervals | 66–73% coverage at ICB/trust before calibration; 75–80% after | 🟡 |
| 4-hour risk accuracy | median error 3.86 pp vs 4.27 pp for the best naive rule | 🟢 |
| 4-hour risk calibration | 80% range covers 82%; Brier skill 54% / 23% | 🟢 |
| Scope | Type 1 only; monthly; no UTC attribution in the 4-hour model | 🟡 documented |

## 2. Implementation verification (automated tests)
- **Ingest:** header variants from different years map to the same schema; all 103 months load; no duplicate provider-months.
- **Contracts:**
  - the processed panel passes
  - an artificial step change and impossible counts are rejected
- **Reclassification:**
  - a synthetic 40% stream-out is detected in the right month with factor 0.60 ± 0.02, and the restated series is continuous
  - genuine smooth growth with seasonality is not flagged
- **Reconciliation:**
  - every method produces coherent forecasts from deliberately incoherent inputs
  - unbiased methods leave already-coherent forecasts unchanged (S G S = S)
  - the shrunk covariance is symmetric positive definite
- **Models:**
  - the Easter algorithm is correct for known years
  - COVID adjustment changes only COVID months
  - seasonal naive uses the daily rate
  - ETS and SARIMA recover a clean seasonal series to < 1% MAPE
- **Outputs:**
  - live forecasts are coherent and intervals ordered and non-negative
  - 4-hour quantiles are monotone in [0, 1] and probabilities valid
  - the chosen model beats seasonal naive at every level
- **Dashboard:** runs end to end (Streamlit AppTest) and responds to region, threshold and measure inputs.
- **Course:** every code block in `learn/` executes.

## 3. Data checks
- **Merger restatement:** for each of the 21 retired codes, predecessor + successor is continuous across the merger month (see `notebooks/01`).
- **North Manchester General moved to Manchester University FT in April 2021.** Its later parent, Pennine Acute, maps to Northern Care Alliance from October 2021. A one-to-one code mapping can't split this, so ICB totals are correct but the two trusts have a step in April 2021. This is inside the COVID window, which the winning attendance model doesn't train on.
- **Recording changes:**
  - 32 detected, with factors 0.54–1.77.
  - Year-on-year jumps above 28% in trust Type 1 series (2023 onwards) fall from 141 trust-months to 42 after restatement. The remainder are candidates for review with trust information teams.
- **Restatement improved accuracy:** attendance MASE averaged over levels went from 0.568 to 0.532, and ICB MAPE from 3.9% to 3.2%.

## 4. Demand forecasts: backtest

### Attendances (ensemble · MinT · post-COVID)
| Level | MASE | MAPE | CRPS (scaled) | 80% cov | 95% cov | Seasonal-naive MASE |
|---|---|---|---|---|---|---|
| England | 0.396 | 1.7% | 0.307 | 88% | 95% | 0.707 |
| Region | 0.459 | 2.1% | 0.335 | 87% | 96% | 0.719 |
| ICB | 0.613 | 3.2% | 0.446 | 77% | 91% | 0.868 |
| Trust | 0.662 | 4.2% | 0.489 | 80% | 92% | 0.886 |

### Emergency admissions (ensemble · MinT · imputed COVID)
| Level | MASE | MAPE | 80% cov (model-based) | Seasonal-naive MASE |
|---|---|---|---|---|
| England | 0.377 | 2.0% | 84% | 0.512 |
| Region | 0.522 | 2.8% | 77% | 0.681 |
| ICB | 0.752 | 5.8% | 67% | 0.962 |
| Trust | 0.882 | 8.5% | 73% | 1.096 |

### Reconciliation (MASE of the winning model under each method)
| Method | Attendances: England / ICB / Trust | Emergency admissions: England / ICB / Trust |
|---|---|---|
| none | 0.441 / 0.627 / 0.665 | 0.415 / 0.768 / 0.885 |
| bottom-up | 0.420 / 0.619 / 0.665 | 0.398 / 0.763 / 0.885 |
| top-down | 0.441 / 0.704 / 0.756 | 0.415 / 0.842 / 0.980 |
| OLS | 0.436 / 0.644 / 0.689 | 0.414 / 0.785 / 0.916 |
| WLS | 0.421 / 0.624 / 0.671 | 0.403 / 0.771 / 0.901 |
| **MinT** | **0.396 / 0.613 / 0.662** | **0.377 / 0.752 / 0.882** |

### COVID treatment
There is no universal winner:
- **Attendances:** the ensemble wins on post-COVID data (0.532 vs 0.591 imputed).
- **Emergency admissions:** imputed history wins (0.633 vs 0.753).
- **SARIMA on its own:** imputation helps (0.563 vs 0.634), because seasonal differencing carries a shock forward a year.
- Selection is part of the backtest.

### Bias
Mean England-level error by origin for emergency admissions is negative at every origin from April 2024 (−0.8% to −2.7%), and positive at the two 2023/early-2024 origins. Admissions per 100 Type 1 attendances fell from 30.0 (2019) to 26.8 (2026). The models extrapolate the 2021–23 recovery. **Finding:** flag to operational teams; don't tune it away. A level-shift term should be considered once the cause is understood.

## 5. Intervals: online calibration test (origins July 2024 to October 2025)

| Measure · option | 80% coverage, England / Region / ICB / Trust | Mean gap from nominal |
|---|---|---|
| Attendances · model-based (chosen) | 96 / 92 / 82 / 83% | 0.109 |
| Attendances · pooled conformal | 96 / 95 / 87 / 87% | 0.153 |
| Emergency admissions · model-based | 79 / 73 / 66 / 73% | 0.142 |
| **Emergency admissions · pooled conformal (chosen)** | 89 / 79 / 75 / 80% | **0.057** |

Finer groupings (per level, per level × horizon band) did worse, because England is a single series.

## 6. 4-hour risk: backtest (7 origins, 10,080 trust-month forecasts)

| Metric | Result |
|---|---|
| MAE of median | **3.86 pp** (last month 4.36, last 3 months 4.27, same month last year 4.99) |
| Mean pinball loss (7 quantiles) | 1.14 pp |
| 80% range (q10–q90) coverage | 74% raw → **82%** conformal |
| 90% range (q05–q95) coverage | 85% raw → **91%** conformal |
| Brier, P(below 60%) | 0.116 vs 0.250 climatology (**skill 54%**) |
| Brier, P(worse than same month last year) | 0.189 vs 0.244 (**skill 23%**) |

**Feature importance (gain):** last month's performance 39%, last 3 months 23%, national pressure 11%. Demand pressure is about 2% (3% including last year's). Recent flow dominates; demand adds little at monthly resolution.

## 7. Limitations
1. Type 1 only. Trust performance against the all-types standard also depends on attributed urgent treatment centre activity (the attribution file is in `data/raw/` for an extension).
2. Monthly resolution: weekly and daily peaks are invisible.
3. Recording-change detection uses thresholds (15%, net < 50%). Smaller or gradual reclassifications are not restated.
4. The intervals assume errors at a given level are exchangeable across origins. A regime change (such as the admissions shift) breaks this.
5. The selection criterion weights the four levels equally. A planner who cares only about trusts might choose differently. Bottom-up is within 0.5% at trust level.
