"""Rolling-origin backtest: every model x every reconciliation method x every hierarchy level.

For each forecast origin we train only on data up to that month, forecast 12 months ahead,
reconcile, and score against what actually happened.

Point accuracy : MASE (scaled by the in-sample seasonal-naive error, so levels are comparable)
Probabilistic  : CRPS and 80% / 95% interval coverage from Gaussian intervals whose covariance comes
                 from the reconciled error covariance S G (h W) G' S'
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from hierarchy import Hierarchy, shrink_cov  # noqa: E402
from models import base_forecast, covid_adjust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ["snaive", "ets", "sarima", "lgbm"]
COMBO = ["ets", "sarima", "lgbm"]  # equal-weight ensemble of these three
RESID_WINDOW = 36   # months of in-sample residuals used for W
VAR_POWER = 0.5
RECON = ["base", "bu", "td", "ols", "wls_struct", "mint_shrink"]


def load(measure: str = "attendances"):
    panel = pd.read_csv(PROC / "trust_month_panel.csv", parse_dates=["month"])
    hier = pd.read_csv(PROC / "hierarchy.csv")
    H = Hierarchy(hier)
    wide = panel.pivot(index="month", columns="trust_code", values=measure).asfreq("MS")
    return H, H.aggregate(wide)


def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def run_origin(args):
    origin, y_all, levels, S_ids, covid_mode, hz = args
    H = _H_CACHE
    train = y_all.loc[:origin]
    if covid_mode == "post":
        train = train.loc["2021-07":]
        adj = train
    else:
        adj = covid_adjust(train, covid_mode)
    actual = y_all.loc[origin + pd.offsets.MonthBegin(): origin + pd.DateOffset(months=hz)]
    h_avail = len(actual)
    recent = y_all.loc[:origin].iloc[-36:]
    mase_scale = np.abs(recent.iloc[12:].to_numpy() - recent.iloc[:-12].to_numpy()).mean(axis=0)

    rows = []
    base = {m: base_forecast(adj, hz, m, levels) for m in MODELS}
    base["combo"] = (sum(base[m][0] for m in COMBO) / len(COMBO), sum(base[m][1] for m in COMBO) / len(COMBO))
    for model, (fc, res) in base.items():
        r = res.iloc[-RESID_WINDOW:].to_numpy()  # recent errors reflect current volatility
        r = r[~np.isnan(r).any(axis=1)]
        W1 = shrink_cov(r)
        for rec in RECON:
            if rec == "base":
                P = np.eye(H.n)
                yhat = fc.to_numpy()
            else:
                G = H.G(rec, residuals=r, history=adj.loc[adj.index[-12]:].to_numpy())
                P = H.S @ G
                yhat = H.reconcile(fc.to_numpy(), G)
            var1 = np.clip(np.diag(P @ W1 @ P.T), 1e-6, None)
            for h in range(1, h_avail + 1):
                sd = np.sqrt(var1 * h ** VAR_POWER)
                yt = actual.iloc[h - 1].to_numpy()
                mu = yhat[h - 1]
                rows.append(pd.DataFrame({
                    "origin": origin, "model": model, "recon": rec, "h": h, "series": S_ids, "level": levels,
                    "actual": yt, "forecast": mu, "sd": sd,
                    "ase": np.abs(yt - mu) / mase_scale,
                    "ape": np.abs(yt - mu) / yt,
                    "crps_scaled": crps_gaussian(yt, mu, sd) / mase_scale,
                    "in80": np.abs(yt - mu) <= norm.ppf(0.9) * sd,
                    "in95": np.abs(yt - mu) <= norm.ppf(0.975) * sd,
                }))
    return pd.concat(rows, ignore_index=True)


_H_CACHE = None


def _init(hier_ids):
    global _H_CACHE
    hier = pd.read_csv(PROC / "hierarchy.csv")
    _H_CACHE = Hierarchy(hier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="attendances")
    ap.add_argument("--covid", default="impute", choices=["impute", "none", "post"])
    ap.add_argument("--first", default="2023-10-01")
    ap.add_argument("--last", default="2025-10-01")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    H, y = load(a.measure)
    origins = pd.date_range(a.first, a.last, freq=f"{a.step}MS")
    jobs = [(o, y, H.levels, H.ids, a.covid, 12) for o in origins]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(H.ids,)) as ex:
        out = pd.concat(list(ex.map(run_origin, jobs)), ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    out.to_parquet(RESULTS / f"backtest_{a.measure}_{a.covid}.parquet", index=False)

    summ = (out.groupby(["model", "recon", "level"])
            .agg(MASE=("ase", "mean"), MAPE=("ape", "mean"), CRPS=("crps_scaled", "mean"),
                 cov80=("in80", "mean"), cov95=("in95", "mean")).reset_index())
    summ.to_csv(RESULTS / f"backtest_summary_{a.measure}_{a.covid}.csv", index=False)
    piv = summ.pivot_table(index=["model", "recon"], columns="level", values="MASE")
    piv = piv[["England", "Region", "ICB", "Trust"]]
    piv["All"] = summ.groupby(["model", "recon"])["MASE"].mean()
    print(piv.round(3).sort_values("All").to_string())


if __name__ == "__main__":
    main()
