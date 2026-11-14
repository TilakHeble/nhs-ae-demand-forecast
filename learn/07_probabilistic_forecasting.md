# 07 · Probabilistic forecasts

A point forecast says "1.51 million attendances in December". A planner also needs to know "how bad could it plausibly be?" Capacity is set for the upper end of the range, not the average. Probabilistic forecasts give a **distribution**, usually summarised as intervals or quantiles.

## 1. Intervals from a model

If forecast errors are roughly Normal with standard deviation σ_h at horizon h:

    80% interval: ŷ ± 1.28 σ_h        95% interval: ŷ ± 1.96 σ_h

σ_h grows with h. How fast it grows is an empirical question. This project first assumed variance ∝ h, which gave intervals that covered 96–100% of outcomes when they promised 80%. They were uselessly wide. Testing variance ∝ √h and using only recent residuals brought attendances close to nominal.

For a **hierarchy**, the reconciled forecast is ỹ = S G ŷ, so its error covariance is S G W G′ S′, where W is the base-forecast error covariance. That is how `src/backtest.py` gives every level its own coherent interval.

## 2. Scoring a distribution

**Coverage.** What share of actual outcomes fall inside the 80% interval? It should be about 80%. Coverage alone isn't enough, because an interval from 0 to infinity always covers.

**Pinball (quantile) loss** for quantile level a:

    L_a(y, q) = a (y − q)        if y ≥ q
                (1 − a)(q − y)   if y < q

It rewards a quantile that is exceeded exactly a of the time. Averaging over many quantile levels approximates the next score.

**CRPS (continuous ranked probability score)** compares the whole predictive CDF with the outcome. Its units are the same as the data. For a point forecast it reduces to MAE, so it's the natural probabilistic version of MAE. For a Normal forecast there is a closed form, used in `src/backtest.py::crps_gaussian`.

**Brier score** for a yes/no event with predicted probability p: mean (p − outcome)². Compare it with the "climatology" Brier, which always predicts the base rate, to get a skill score: 1 − Brier / Brier_climatology.

## 3. Conformal calibration

When model-based intervals are mis-calibrated, **split conformal prediction** fixes them with almost no assumptions:

1. On past forecasts, compute standardised errors |z| = |y − ŷ| / σ̂.
2. The 80th percentile of |z| (with a finite-sample correction) is the multiplier that would have given 80% coverage.
3. New interval: ŷ ± multiplier × σ̂.

For quantile models, **conformalised quantile regression** shifts each predicted quantile by the empirical quantile of its own residuals.

**Evaluate conformal methods online.** At each origin, calibrate only on forecasts whose outcomes were already known by then. `src/calibrate.py` does this and found a trade-off. A multiplier per level and horizon band adapts most but is noisy for England, which is a single series. One pooled multiplier was most reliable for emergency admissions. For attendances the original model-based intervals were already the best calibrated, so they were kept.

## Try it: coverage and CRPS from the backtest

```python
import pandas as pd, numpy as np
b = pd.read_parquet("results/backtest_attendances_post.parquet")  # created by `make backtest`
b = b[(b.model == "combo") & (b.recon == "mint_shrink")]
print(b.groupby("level")[["in80", "in95", "crps_scaled"]].mean().round(3))
```

## 4. The 4-hour risk model: quantiles to probabilities

`src/four_hour.py` fits LightGBM with the **quantile objective** at 7 levels (5th…95th percentile) and interpolates a CDF between them to read off P(performance < threshold). Things to check in your own work:

- **Quantile crossing.** Separately trained quantile models can cross, for example q90 < q75. Sort them.
- **Calibration** of the derived probabilities. Bin the predictions and compare predicted with observed event rates (`results/four_hour_calibration.csv`).
- **Tails.** Beyond the outermost quantile you need an assumption. The repo extends Gaussian tails so probabilities still rank trusts sensibly.
