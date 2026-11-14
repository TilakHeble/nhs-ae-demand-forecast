"""Hierarchy structure and forecast reconciliation (bottom-up, top-down, OLS, WLS, MinT).

Notation (Hyndman & Athanasopoulos, FPP3 ch. 11):
  y_t  = S b_t        all n series are S (n x m) times the m bottom-level series
  ŷ_h  = base forecasts for all n series (incoherent: trusts don't add up to England)
  ỹ_h  = S G ŷ_h      reconciled forecasts; G picks how to blend levels

  Bottom-up:  G = [0 | I_m]
  Top-down:   G = [p | 0]     p = historical proportions of each trust in England
  OLS:        G = (S'S)^-1 S'
  WLS:        G = (S'W⁻¹S)^-1 S'W⁻¹  with W = diag(number of trusts under each node)  ("structural scaling")
  MinT:       same formula with W = shrunk covariance of in-sample base-forecast errors (Wickramasuriya et al. 2019)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Hierarchy:
    def __init__(self, hier: pd.DataFrame):
        h = hier.sort_values(["region_name", "icb_name", "trust_code"]).reset_index(drop=True)
        self.bottom = h["trust_code"].tolist()
        self.regions = sorted(h["region_name"].unique())
        self.icbs = h.drop_duplicates("icb_code").sort_values(["region_name", "icb_name"])["icb_code"].tolist()
        rows, ids, levels = [], [], []

        rows.append(np.ones(len(h)))
        ids.append("England")
        levels.append("England")
        for r in self.regions:
            rows.append((h["region_name"] == r).to_numpy(float))
            ids.append(f"R:{r}")
            levels.append("Region")
        for c in self.icbs:
            rows.append((h["icb_code"] == c).to_numpy(float))
            ids.append(f"I:{c}")
            levels.append("ICB")
        for t in self.bottom:
            rows.append((h["trust_code"] == t).to_numpy(float))
            ids.append(f"T:{t}")
            levels.append("Trust")

        self.S = np.vstack(rows)
        self.ids = ids
        self.levels = np.array(levels)
        self.n, self.m = self.S.shape
        names = {f"R:{r}": r for r in self.regions}
        names.update({f"I:{c}": n for c, n in h.drop_duplicates("icb_code")[["icb_code", "icb_name"]].to_numpy()})
        names.update({f"T:{t}": n for t, n in h[["trust_code", "trust_name"]].to_numpy()})
        names["England"] = "England"
        self.names = names
        self.meta = h

    def aggregate(self, bottom_wide: pd.DataFrame) -> pd.DataFrame:
        """bottom_wide: index=month, columns=trust codes -> all n series (columns = ids)."""
        b = bottom_wide[self.bottom].to_numpy()
        return pd.DataFrame(b @ self.S.T, index=bottom_wide.index, columns=self.ids)

    # ---------- reconciliation ----------
    def G(self, method: str, residuals: np.ndarray | None = None, history: np.ndarray | None = None) -> np.ndarray:
        S, n, m = self.S, self.n, self.m
        if method == "bu":
            return np.hstack([np.zeros((m, n - m)), np.eye(m)])
        if method == "td":
            # average historical proportions of each trust in the England total (Gross-Sohl method A)
            if history is None:
                raise ValueError("top-down needs history (T x n)")
            p = (history[:, n - m:] / history[:, [0]]).mean(axis=0)
            G = np.zeros((m, n))
            G[:, 0] = p
            return G
        if method == "ols":
            W = np.eye(n)
        elif method == "wls_struct":
            W = np.diag(S.sum(axis=1))
        elif method == "wls_var":
            if residuals is None:
                raise ValueError("wls_var needs residuals")
            W = np.diag(np.nanvar(residuals, axis=0))
        elif method == "mint_shrink":
            if residuals is None:
                raise ValueError("mint_shrink needs residuals")
            W = shrink_cov(residuals)
        else:
            raise ValueError(method)
        Winv = np.linalg.pinv(W)
        return np.linalg.solve(S.T @ Winv @ S, S.T @ Winv)

    def reconcile(self, base: np.ndarray, G: np.ndarray) -> np.ndarray:
        """base: (H x n) base forecasts -> (H x n) coherent forecasts."""
        return base @ (self.S @ G).T


def shrink_cov(resid: np.ndarray) -> np.ndarray:
    """Schäfer-Strimmer shrinkage of the residual covariance towards its diagonal (as in MinT-shrink)."""
    e = resid[~np.isnan(resid).any(axis=1)]
    T, n = e.shape
    e = e - e.mean(axis=0)
    cov = e.T @ e / T
    sd = np.sqrt(np.diag(cov))
    sd[sd == 0] = 1e-9
    corr = cov / np.outer(sd, sd)
    es = e / sd
    w = es ** 2
    # variance of each sample correlation: Var(r_ij) = T/(T-1)^3 * sum_t (w_tij - mean_ij)^2, w_tij = z_ti z_tj
    v = (w.T @ w) / T - ((es.T @ es) / T) ** 2
    v = v * T ** 2 / (T - 1) ** 3
    np.fill_diagonal(v, 0)
    off = corr.copy()
    np.fill_diagonal(off, 0)
    lam = float(np.clip(v.sum() / max((off ** 2).sum(), 1e-12), 0, 1))
    shrunk = lam * np.diag(np.diag(corr)) + (1 - lam) * corr
    return shrunk * np.outer(sd, sd)
