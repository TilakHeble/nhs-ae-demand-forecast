# 08 · Hierarchical forecasting and reconciliation

## 1. The problem: incoherence

Forecast each node of a hierarchy separately and the numbers won't add up:

    Σ trust forecasts in an ICB  ≠  ICB forecast
    Σ regional forecasts         ≠  England forecast

Different teams then plan with different numbers. **Coherent** forecasts respect the aggregation constraints exactly.

## 2. The summing matrix

Stack every series into one vector y_t (n = 164 here) and the bottom level into b_t (m = 120 trusts). Then

    y_t = S b_t

where S is n × m. Row i of S has a 1 for every trust under node i. A toy example with 2 regions and 4 trusts:

```
          T1 T2 T3 T4
England [ 1  1  1  1 ]
R1      [ 1  1  0  0 ]
R2      [ 0  0  1  1 ]
T1      [ 1  0  0  0 ]
...     (identity for the trusts)
```

## 3. Reconciliation as a linear map

Take base forecasts ŷ for all n series (incoherent). Any linear reconciliation is

    ỹ = S G ŷ

G (m × n) maps all the base forecasts to bottom-level forecasts, and S adds them back up. **Any G gives coherent forecasts**, because the result lies in the column space of S. The methods differ only in G:

| Method | G | Intuition |
|---|---|---|
| Bottom-up | [0 \| I_m] | trust forecasts only. Unbiased, but noisy at the top |
| Top-down | [p \| 0] | England forecast × historical shares p. Smooth, but blind to local trends |
| OLS | (S′S)⁻¹S′ | orthogonal projection onto the coherent subspace |
| WLS (structural) | (S′Λ⁻¹S)⁻¹S′Λ⁻¹, Λ = diag(number of trusts under each node) | aggregates get less weight per unit of size |
| **MinT** | (S′W⁻¹S)⁻¹S′W⁻¹, W = covariance of base-forecast errors | **minimises the total variance of the reconciled errors** among unbiased linear methods |

**MinT** (Wickramasuriya, Athanasopoulos & Hyndman, 2019) is a generalised least squares projection. Series with large, correlated errors are trusted less. The full W (164 × 164) can't be estimated reliably from about 50 residuals, so it is **shrunk** towards its diagonal (Schäfer–Strimmer). See `src/hierarchy.py::shrink_cov`.

## Try it: reconcile a toy hierarchy by hand

```python
import numpy as np
S = np.array([[1,1,1,1],[1,1,0,0],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], float)
yhat = np.array([105, 48, 50, 22, 24, 27, 26], float)   # base forecasts: 22+24+27+26 = 99 ≠ 105
print("incoherence at the top:", yhat[0] - yhat[3:].sum())

G_bu  = np.hstack([np.zeros((4, 3)), np.eye(4)])
G_ols = np.linalg.solve(S.T @ S, S.T)
W = np.diag([9.0, 4, 4, 1, 1, 1, 1])                      # say England's base forecast is the noisiest
Wi = np.linalg.inv(W)
G_wls = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
for name, G in {"bottom-up": G_bu, "OLS": G_ols, "WLS": G_wls}.items():
    y = S @ G @ yhat
    print(f"{name:9s}", y.round(1), " coherent:", np.isclose(y[0], y[3:].sum()))
```

The output shows bottom-up ignoring England's 105 entirely, OLS splitting the difference, and WLS leaning towards whichever series has the smaller error variance.

## Try it on the real hierarchy

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from hierarchy import Hierarchy
hier = pd.read_csv("data/processed/hierarchy.csv")
H = Hierarchy(hier)
print(H.S.shape, "→", dict(pd.Series(H.levels).value_counts()))
fc = pd.read_csv("results/forecast_attendances.csv")
gap = fc[fc.level == "Trust"].groupby("month").forecast.sum() - fc[fc.level == "England"].set_index("month").forecast
print("max |Σ trusts − England| =", float(gap.abs().max()))
```

## 4. When does each method win?

From this project's backtest and the literature:

- **Top-down** loses at the bottom whenever trusts have different trends. Here it was consistently the worst method.
- **Bottom-up** is strong when the bottom-level series are long and clean, and was the best partner for a single SARIMA.
- **MinT** wins when base forecasts at different levels carry complementary information, which is the case with a diverse ensemble. It improved England emergency-admission MASE by 9% over the unreconciled forecasts.
- MinT depends on a decent W. With very short residual histories, WLS (structural) is a safe fallback that needs no estimation.

**Reconciliation doesn't only make numbers add up. It usually makes them more accurate**, because each level borrows strength from the others.
