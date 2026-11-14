"""Produce the live 12-month forecast for every trust, ICB, region and England.

Picks the model + reconciliation method with the lowest backtest MASE (averaged over the four
hierarchy levels), refits on all data to the latest month, and writes coherent forecasts with
80% and 95% intervals.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import COMBO, RESID_WINDOW, VAR_POWER, load
from .config import RESULTS
from .hierarchy import shrink_cov
from .models import base_forecast, covid_adjust


def best_config(measure: str) -> tuple[str, str, str]:
    """Lowest average-over-levels MASE across every COVID treatment that has been backtested."""
    frames = []
    for f in RESULTS.glob(f"backtest_summary_{measure}_*.csv"):
        d = pd.read_csv(f)
        d["covid"] = f.stem.rsplit("_", 1)[-1]
        frames.append(d)
    s = pd.concat(frames)
    s = s[s["model"] != "snaive"]
    model, recon, covid = s.groupby(["model", "recon", "covid"])["MASE"].mean().idxmin()
    return model, recon, covid


def interval_bounds(out: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Gaussian intervals, widened by the conformal multipliers if src/calibrate.py chose them."""
    import json
    out = out.assign(mult80=norm.ppf(0.9), mult95=norm.ppf(0.975), interval_method="model-based")
    meth_file = RESULTS / "interval_method.json"
    if meth_file.exists() and measure in (meth := json.loads(meth_file.read_text())):
        keys = meth[measure]["keys"]
        if keys is not None:
            mult = pd.read_csv(RESULTS / f"interval_multipliers_{measure}.csv")
            out = out.drop(columns=["mult80", "mult95"])
            out["band"] = out.h.map(lambda h: "1-3" if h <= 3 else ("4-6" if h <= 6 else "7-12"))
            out = (out.merge(mult, on=keys, how="left") if keys
                   else out.assign(mult80=mult.mult80.iloc[0], mult95=mult.mult95.iloc[0]))
            out = out.drop(columns=["band"])
        out["interval_method"] = meth[measure]["method"]
    for q in ["80", "95"]:
        out[f"lo{q}"] = (out.forecast - out[f"mult{q}"] * out.sd).clip(lower=0)
        out[f"hi{q}"] = out.forecast + out[f"mult{q}"] * out.sd
    return out.drop(columns=["mult80", "mult95"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="attendances")
    ap.add_argument("--horizon", type=int, default=12)
    a = ap.parse_args()

    H, y = load(a.measure)
    model, recon, covid = best_config(a.measure)
    adj = y.loc["2021-07":] if covid == "post" else covid_adjust(y, covid)
    if model == "combo":
        parts = [base_forecast(adj, a.horizon, m, H.levels) for m in COMBO]
        fc = sum(p[0] for p in parts) / len(parts)
        res = sum(p[1] for p in parts) / len(parts)
    else:
        fc, res = base_forecast(adj, a.horizon, model, H.levels)
    r = res.iloc[-RESID_WINDOW:].to_numpy()
    r = r[~np.isnan(r).any(axis=1)]
    if recon == "base":
        P, yhat = np.eye(H.n), fc.to_numpy()
    else:
        G = H.G(recon, residuals=r, history=adj.iloc[-12:].to_numpy())
        P, yhat = H.S @ G, H.reconcile(fc.to_numpy(), G)
    var1 = np.clip(np.diag(P @ shrink_cov(r) @ P.T), 1e-6, None)

    rows = []
    for h in range(1, a.horizon + 1):
        sd = np.sqrt(var1 * h ** VAR_POWER)
        mu = yhat[h - 1]
        rows.append(pd.DataFrame({
            "month": fc.index[h - 1], "h": h, "series": H.ids, "level": H.levels,
            "name": [H.names[i] for i in H.ids], "forecast": mu, "sd": sd,
        }))
    out = interval_bounds(pd.concat(rows, ignore_index=True), a.measure)
    out["model"], out["recon"], out["covid"] = model, recon, covid
    out.to_csv(RESULTS / f"forecast_{a.measure}.csv", index=False)

    hist = y.stack().rename("actual").reset_index()
    hist.columns = ["month", "series", "actual"]
    hist.to_csv(RESULTS / f"history_{a.measure}.csv", index=False)

    eng = out[out.series == "England"]
    check = out[out.level == "Trust"].groupby("month").forecast.sum().to_numpy() - eng.forecast.to_numpy()
    print(f"{a.measure}: model={model}, recon={recon}, covid={covid}; "
          f"max coherence gap (trusts vs England) = {np.abs(check).max():.6f}")
    print(eng.set_index("month")[["forecast", "lo80", "hi80"]].round(0).to_string())


if __name__ == "__main__":
    main()
