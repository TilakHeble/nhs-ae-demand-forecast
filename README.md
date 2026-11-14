# NHS A&E Demand Forecast: forecasts that add up from trust to England

**Coherent 12-month forecasts of A&E demand for every acute trust, ICB and region in England, and the national total**, built from NHS England's official monthly A&E statistics.

> Project 6a in my one-project-per-model portfolio. **Model family: time series** (ETS, SARIMA, global gradient boosting, ensembles) **and hierarchical forecast reconciliation** (bottom-up, top-down, OLS, WLS, MinT).

---

## The problem

NHS capacity planning happens at several levels at once:

- trusts plan beds and rotas
- Integrated Care Boards (ICBs) plan system flow
- regions and NHS England plan winter funding and escalation

If each level forecasts on its own, the numbers don't agree: the trust forecasts in an ICB won't add up to that ICB's forecast. People end up arguing about whose number is right instead of planning.

This project forecasts **all 164 series in the hierarchy** and **reconciles** them, so every level adds up exactly and each level gets more accurate in the process.

```
England (1)
└── Regions (7)
    └── ICBs (36)
        └── Acute trusts with a Type 1 A&E (120)
```

## Data

| Item | Detail |
|---|---|
| Source | NHS England, *A&E Attendances and Emergency Admissions* monthly situation reports (Accredited Official Statistics) |
| Period | April 2018 to October 2026 (103 months, every monthly CSV) |
| Measures | **Type 1 (major) A&E attendances**, including booked appointments; **emergency admissions via Type 1 A&E**. 4-hour breaches are kept for later work |
| Hierarchy | NHS England *Provider to ICB mapping*, October 2026 |

### The messy real-world parts (this is where most of the work went)

1. **Column names change across eight years of files.** For example "A&E attendances Type 1" vs "Number of A&E attendances Type 1", plus stray blank columns, a TOTAL row and trailing spaces. `src/ingest.py` maps every variant onto one schema.
2. **Trust mergers.** 21 Type 1 providers in the files no longer exist under their old code. Ipswich and Colchester became ESNEFT, Poole and Bournemouth became UH Dorset, North Bristol became Bristol NHS FT in 2026, and so on. Left alone, the old series simply stop and the successor trusts look as if their demand doubled overnight. Each retired code is mapped to its successor, and every pair was checked by confirming the combined series is continuous across the merger month. The full list is in [`data/processed/trust_mergers.csv`](data/processed/trust_mergers.csv).
3. **COVID-19.** April 2020 to June 2021 is not normal demand. Two treatments are compared in the backtest:
   - `impute`: replace those months with a counterfactual (2019 same month × each series' recovery ratio)
   - `post`: train only on July 2021 onwards
4. **Calendar effects.** All models forecast the **daily rate** (monthly total ÷ days in month) on a log scale. That removes the February effect and makes seasonality multiplicative. Easter, which moves between March and April, is a feature in the boosted model.

## Methods

**Base models.** Each one is fitted to all 164 series:

| Model | Detail |
|---|---|
| Seasonal naive | Same month last year. Every model has to beat this |
| ETS | Holt-Winters with damped additive trend and additive seasonality, on the log daily rate |
| SARIMA | The "airline" model ARIMA(0,1,1)(0,1,1)₁₂ on the log daily rate |
| Global LightGBM | **One model trained across all 164 series**, direct multi-horizon (h = 1…12). Features are scale-free: same month last year, last-3-month level, year-on-year momentum, horizon, month, Easter, days in month, hierarchy level |
| Ensemble | Equal-weight average of ETS, SARIMA and LightGBM |

**Reconciliation.** Every base model is combined with each of these methods. Notation from Hyndman & Athanasopoulos, *FPP3*, ch. 11: ỹ = S G ŷ.

| Method | G |
|---|---|
| Bottom-up | Use only the trust forecasts |
| Top-down | England forecast × historical trust shares |
| OLS | (S′S)⁻¹S′ |
| WLS (structural) | W = number of trusts under each node |
| **MinT-shrink** | W = shrinkage estimate of the base-forecast error covariance (Wickramasuriya, Athanasopoulos & Hyndman, 2019), implemented from scratch in [`src/hierarchy.py`](src/hierarchy.py) |

**Evaluation.**
- Rolling-origin backtest: 9 origins from October 2023 to October 2025, forecasting 12 months ahead each time. Each model is trained only on data available at its origin.
- Point accuracy: **MASE** (scaled by the in-sample seasonal-naive error, so trusts and England are comparable) and MAPE.
- Probabilistic accuracy: **CRPS**, plus 80% and 95% interval coverage. The intervals come from the reconciled error covariance S G W G′ S′.

That gives 5 models × 6 reconciliation methods × 2 COVID treatments × 2 measures × 164 series × 9 origins × 12 horizons, about **2.1 million scored forecasts**.

## Results

### Type 1 A&E attendances

Winning configuration: **Ensemble + MinT-shrink, trained on post-COVID data.**

| Level | MASE | MAPE | CRPS (scaled) | 80% PI coverage | 95% PI coverage | Seasonal-naive MASE |
|---|---|---|---|---|---|---|
| England | **0.439** | 1.8% | 0.338 | 87% | 97% | 0.700 |
| Region | **0.501** | 2.3% | 0.367 | 85% | 95% | 0.719 |
| ICB | **0.658** | 3.9% | 0.491 | 76% | 89% | 0.898 |
| Trust | **0.675** | 4.8% | 0.508 | 80% | 91% | 0.905 |

### Emergency admissions via Type 1 A&E

Winning configuration: **Ensemble + MinT-shrink, trained on COVID-imputed history.**

| Level | MASE | MAPE | 80% PI coverage | Seasonal-naive MASE |
|---|---|---|---|---|
| England | **0.377** | 2.0% | 84% | 0.512 |
| Region | **0.522** | 2.8% | 77% | 0.681 |
| ICB | **0.752** | 5.8% | 67% | 0.962 |
| Trust | **0.882** | 8.5% | 73% | 1.096 |

### What I learned

- **The ensemble wins for both measures.** The three models make different mistakes, so averaging them works better than any single model.
- **MinT beats bottom-up and top-down at every level once you use an ensemble.** Top-down is consistently the worst: splitting England by historical shares misses trust-level trends. For emergency admissions, MinT cut England-level MASE by 9% compared with the unreconciled base forecasts (0.415 → 0.377).
- **There is no universal COVID fix.** For attendances, dropping the COVID years was best. Demand patterns since 2021 differ enough that older data hurts more than it helps. For emergency admissions, which have more trust-level noise, the extra years of imputed history were worth keeping. The same choice tells a single SARIMA the opposite: imputation helped it (0.591 vs 0.677). This is why the backtest searches COVID treatment alongside the model.
- **Interval calibration needs testing.** The first version scaled variance linearly with the horizon, and the 80% intervals covered 96–100% of outcomes, which is far too wide to plan with. Using recent residuals (the last 36 months) and √h variance growth gets attendances close to the nominal 80%. Emergency-admission intervals are still too narrow at ICB and trust level: listed under next steps.
- **A global LightGBM is not automatically better.** On its own it ranked mid-table. Its value here is as a diverse member of the ensemble.

### Live forecast: November 2026 to October 2027

England Type 1 attendances are forecast to grow about **3.7% year on year**, in line with the recent trend. February 2027 is forecast at 1.36m (80% interval 1.33m–1.39m) and October 2027 at 1.59m. The dashboard's winter-outlook tab ranks trusts by forecast December–February demand against last winter.

## Dashboard

`streamlit run app/streamlit_app.py`

- **Forecast:** pick a region, ICB or trust and see the reconciled forecast with 80% and 95% intervals
- **Winter outlook:** trusts ranked by forecast change in December–February demand
- **Model comparison:** every model × reconciliation method at every level (MASE, CRPS, MAPE, coverage), plus accuracy by horizon
- **How reconciliation works:** plain-English explainer

## Run it yourself

```bash
pip install -r requirements-dev.txt
make data       # build the trust-month panel + hierarchy from data/raw
make backtest   # ~25 minutes on 2 cores
make forecast   # refit the winning configuration, write results/forecast_*.csv
make app
```

## Repository structure

```
data/raw/           NHS England monthly CSVs (bundled), provider→ICB mapping, attribution file
data/processed/     trust_month_panel.csv · hierarchy.csv · trust_mergers.csv
src/ingest.py       read and harmonise eight years of changing file formats
src/build_panel.py  merger restatement, hierarchy build
src/hierarchy.py    summing matrix S, BU/TD/OLS/WLS/MinT reconciliation, shrinkage covariance
src/models.py       COVID adjustment, seasonal naive, ETS, SARIMA, global LightGBM
src/backtest.py     rolling-origin evaluation (MASE, MAPE, CRPS, coverage)
src/forecast.py     picks the best configuration from the backtest and forecasts 12 months
src/report.py       compact result tables
app/                Streamlit dashboard
results/            forecasts, history, backtest summaries
```

## Limitations and next steps

- **History is restated on today's structure.** One trust move can't be fully separated: North Manchester General moved from Pennine Acute to Manchester University FT in April 2021, before the rest of Pennine joined Northern Care Alliance. ICB-level totals are unaffected because both trusts sit in Greater Manchester.
- Emergency-admission intervals undercover at ICB and trust level. Next step: horizon-specific empirical calibration or conformal intervals.
- **4-hour performance risk.** Turn the demand forecast into a probability that each trust misses the 4-hour standard this winter.
- `learn/` folder: time series from basics to hierarchical reconciliation, as in my earlier projects.
- Monthly data hides weekly and daily peaks. ECDS daily data would allow forecasts for staffing rotas.

---
*Data: NHS England, A&E Attendances and Emergency Admissions, © Crown copyright, Open Government Licence v3.0.*
