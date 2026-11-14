# Executive summary

**NHS A&E demand forecast: coherent forecasts from trust to England, and 4-hour risk by trust**

## The question
How much Type 1 (major) A&E demand should each of England's 120 acute trusts, 36 ICBs and 7 regions plan for over the next 12 months? The forecasts need to add up exactly between levels. And which trusts are most likely to see 4-hour performance slip this winter?

## Why it matters
Capacity planning happens at every level at once. When each level forecasts separately, the trust forecasts don't add up to the ICB's, and the ICB forecasts don't add up to the region's. Plans then conflict before winter even starts. Forecasts nobody can check against history (no backtest, no honest uncertainty) get overridden by gut feel.

## What was done
- **Data.** Every NHS England monthly A&E return, April 2018 to October 2026 (103 months), plus the October 2026 provider-to-ICB mapping.
- **History restated onto today's NHS:**
  - 21 trust mergers, each checked for continuity
  - **32 Type 1 recording changes at 25 trusts.** Patients streamed to (or back from) a co-located urgent treatment centre move between "Type 1" and "Other" in the returns. This would otherwise look like demand collapsing or doubling overnight. An automated data contract found it.
- **Forecasting.** Five models (seasonal naive, ETS, SARIMA, a global LightGBM, an ensemble), each combined with six reconciliation methods and two COVID treatments. They were scored in a 9-origin rolling backtest on about 2.1 million forecasts.
- **Uncertainty.** Prediction intervals were checked for coverage, then calibrated with split-conformal prediction where needed, chosen by an online test.
- **4-hour risk.** A quantile LightGBM turns the demand forecast and each trust's recent flow into a probability of missing any 4-hour threshold.

## Results

| | |
|---|---|
| **Accuracy (attendances)** | Ensemble + MinT reconciliation. MAPE **1.7%** England, **2.1%** region, **3.2%** ICB, **4.2%** trust. MASE 0.40 / 0.46 / 0.61 / 0.66, against 0.71 / 0.72 / 0.87 / 0.89 for "same month last year" |
| **Reconciliation** | MinT is the most accurate method at every level, and **10% more accurate at England level** than forecasting without reconciliation. Top-down splitting is the worst: 12–14% worse at ICB and trust level |
| **Recording-change fix** | Restating the 32 recording changes cut ICB-level error from 3.9% to 3.2% MAPE |
| **Emergency admissions** | Ensemble + MinT. MAPE 2.0% England, 8.5% trust. They have come in **1–3% below forecast at every origin since April 2024**. Admissions per 100 attendances fell from 30 (2019) to about 27 (2026). That is a real shift for operational teams to explain, not noise |
| **Intervals** | Emergency-admission 80% intervals originally covered 66–73% at ICB and trust level. After conformal calibration: 75–80% |
| **4-hour risk** | Median error 3.9 percentage points, against 4.3 for the best simple rule. Brier skill 54% for P(below 60%). Calibrated 80% ranges cover 82% of outcomes |

## The forecast
- England Type 1 attendances: **+3.7%** over November 2026 to October 2027, about 18.2 million.
- Winter 2026/27: median trust Type 1 4-hour performance about **59%**.
  - 69 of 120 trusts are more likely than not to fall below 60%.
  - 56 are more likely than not to do worse than last winter.
- These are Type 1 figures. The national 82% objective covers all A&E types.

## Recommendations
1. **Plan from one coherent set of numbers.** Reconciled forecasts are more accurate as well as consistent.
2. **Keep a register of recording changes** alongside the merger register. Without it, any Type 1 trend analysis at trust level is unreliable.
3. **Investigate the fall in admissions per attendance** before relying on admissions-based bed models.
4. **Target winter support on trusts with persistently weak recent flow.** Recent performance, not demand, is the strongest predictor of 4-hour risk.

## Limits
- The data is monthly and Type 1 only.
- The 4-hour model ignores attributed urgent treatment centre activity.
- One trust move (North Manchester, April 2021) can't be split by site.
- See the [validation report](validation_report.md).
