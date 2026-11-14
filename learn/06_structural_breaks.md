# 06 · Structural breaks: COVID and mergers

Real data has breaks: the relationship between past and future changes abruptly. Text-book series rarely do. Two kinds dominate NHS data.

## 1. Interventions: COVID-19

In April 2020 Type 1 attendances fell by roughly half almost overnight, then recovered unevenly through 2021. A model fitted naively treats that as normal behaviour:

- **ETS** follows the level down and then has to climb back.
- **SARIMA** carries the shock forward through the seasonal difference (y_t − y_{t−12}), so a 2020 dip contaminates the 2021 forecast.
- **ML lags** ("same month last year") are garbage for 12 months after the shock.

Options:

| Approach | How | Trade-off |
|---|---|---|
| **Truncate** | train only on data after recovery (July 2021 onwards here) | clean, but less history (about 5 years) |
| **Impute** | replace the break with a counterfactual of what demand would have been | keeps history, but you are modelling your own guess |
| **Dummy variables** | add a 0/1 regressor for the COVID months (SARIMAX) | statistically neat, but needs a regression framework and the right shape for the dummy |
| **Downweight** | give COVID months small weights in a loss function | works for ML models |

`src/nhsae/models.py::covid_adjust` imputes: each COVID month = same month in 2019 × a ratio that blends linearly from the 2019 level towards the series' own 2021–22 level.

**What the backtest found.** Truncation won for attendances. Imputation won for emergency admissions and for SARIMA on its own. **There is no universal answer.** Treat the choice as a hyper-parameter and let a rolling-origin backtest decide, which is what `src/nhsae/forecast.py` does.

## 2. Organisational breaks: trust mergers

When two trusts merge, the old codes stop reporting and the new (or surviving) code suddenly reports both sites. Between 2018 and 2026, 21 Type 1 providers disappeared this way.

If you ignore this:

- the old series end abruptly (forecasting them is meaningless)
- the successor series show a fake doubling, which a model reads as explosive growth

**Fix: restate history on today's structure.** Map each retired code to its successor and add their histories together:

```python
import pandas as pd
m = pd.read_csv("data/processed/trust_mergers.csv")
print(m[["retired_code", "successor_code", "first_combined_month", "note"]].head(8).to_string(index=False))
```

**Always validate a merger mapping.** If it's right, predecessor + successor is continuous across the merger month. The merger table in `src/nhsae/panel.py` was built by checking every pair this way. One awkward case is documented: North Manchester General moved from Pennine Acute to Manchester University FT six months before the rest of Pennine joined the Northern Care Alliance. A one-to-one mapping can't split a single trust's history by site, so ICB totals are right but those two trusts have a small step in April 2021.

## 3. Recording changes: the break a data contract found

After the mergers were mapped, an automated check (no trust's Type 1 daily rate may jump more than 60% in a month)
still failed for two trusts that hadn't merged. They had changed **how they record** attendances. When a trust opens
a co-located urgent treatment centre and streams minor cases there, those patients move from "Type 1" to "Other" in
the returns. Northern Lincolnshire & Goole's Type 1 count fell 42% overnight in May 2024, and its "Other" count rose by
almost exactly the same amount.

`src/nhsae/reclass.py` detects these steps (Type 1 moves by more than 15%, "Other" moves the opposite way, and the
two largely cancel). It found 32 at 25 trusts and restates earlier history onto each trust's current basis.

```python
import pandas as pd
ev = pd.read_csv("data/processed/type1_reclassifications.csv", parse_dates=["month"])
print(len(ev), "recording changes at", ev.trust_code.nunique(), "trusts")
print(ev.sort_values("t1_change")[["trust_name", "month", "t1_change", "factor"]].head(5).round({"t1_change": 2, "factor": 2}).to_string(index=False))
```

The lesson: **write data contracts that encode what you believe about the data, and let them fail.** This one was
written to catch unmapped mergers and found a second, bigger problem.

## 4. Detecting breaks you don't know about

```python
import pandas as pd, numpy as np
p = pd.read_csv("data/processed/trust_month_panel.csv", parse_dates=["month"])
w = p.pivot(index="month", columns="trust_code", values="attendances")
rate = w.div(w.index.days_in_month, axis=0)
yoy = np.log(rate).diff(12).loc["2023-01":]             # year-on-year log change
flag = (yoy.abs() > 0.25)                                 # >~28% jump vs same month last year
print("trust-months with suspicious year-on-year jumps:", int(flag.sum().sum()))
print(flag.sum().sort_values(ascending=False).head(5))
```

Run on the reported figures this finds 141 suspicious trust-months; on the restated series, 42. What is left is worth investigating one by one. Large year-on-year jumps after the COVID period usually mean something organisational: a site opened or closed, a service was reconfigured, or a reporting change. Check them before you model. In an NHS team you'd ask the trust's information team. In a portfolio project, you document it.
