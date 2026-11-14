"""Calibrate prediction intervals with split-conformal scaling, chosen by an online backtest.

The model-based intervals (Gaussian, covariance S G W G' S') assume future errors look like the
in-sample residuals. For emergency admissions they don't: the 80% intervals covered only ~66-73% of
outcomes at ICB and trust level. Split conformal prediction fixes this without assuming a distribution:

  1. take the backtest's standardised errors  z = (actual - forecast) / sd
  2. the empirical q-quantile of |z| is the multiplier that would have made the interval cover q
  3. new interval = forecast ± multiplier × sd

How finely to estimate the multiplier is a bias-variance trade-off. A multiplier per level and horizon
band adapts more, but England has only one series, so its quantile is very noisy. We test four options
**online**, the way it would run in production: at each origin the multipliers are estimated only
from earlier forecasts whose target month had already happened. The option whose coverage is closest
to the nominal 80% / 95% wins for each measure, and "keep the model-based intervals" is one of the options.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import RESULTS  # noqa: E402
from forecast import best_config  # noqa: E402

LEVELS = ["England", "Region", "ICB", "Trust"]
QS = {"80": 0.80, "95": 0.95}
OPTIONS = {  # name -> grouping keys (None = model-based Gaussian intervals, no conformal step)
    "model-based": None,
    "conformal: level x horizon band": ["level", "band"],
    "conformal: per level": ["level"],
    "conformal: pooled": [],
}


def band(h: int) -> str:
    return "1-3" if h <= 3 else ("4-6" if h <= 6 else "7-12")


def standardised_errors(measure: str) -> pd.DataFrame:
    model, recon, covid = best_config(measure)
    b = pd.read_parquet(RESULTS / f"backtest_{measure}_{covid}.parquet")
    b = b[(b.model == model) & (b.recon == recon)].copy()
    b["absz"] = (b.actual - b.forecast).abs() / b.sd
    b["band"] = b.h.map(band)
    b["target"] = [o + pd.DateOffset(months=int(h)) for o, h in zip(b.origin, b.h)]
    return b


def fit_multipliers(cal: pd.DataFrame, keys: list[str] | None) -> pd.DataFrame:
    """Finite-sample conformal quantile ceil((n+1)q)/n of |z| within each group."""
    if keys is None:
        return pd.DataFrame({"mult80": [norm.ppf(0.9)], "mult95": [norm.ppf(0.975)]})
    groups = cal.groupby(keys) if keys else [((), cal)]
    rows = []
    for k, g in groups:
        n = len(g)
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for lbl, q in QS.items():
            row[f"mult{lbl}"] = float(np.quantile(g.absz, min(1.0, np.ceil((n + 1) * q) / n)))
        rows.append(row)
    return pd.DataFrame(rows)


def apply(d: pd.DataFrame, mult: pd.DataFrame, keys: list[str] | None) -> pd.DataFrame:
    if not keys:
        return d.assign(mult80=float(mult.mult80.iloc[0]), mult95=float(mult.mult95.iloc[0]))
    return d.merge(mult, on=keys, how="left")


def online_coverage(b: pd.DataFrame, keys) -> pd.DataFrame:
    out = []
    for o in sorted(b.origin.unique())[3:]:
        cal = b[(b.origin < o) & (b.target <= o)]
        t = apply(b[b.origin == o], fit_multipliers(cal, keys), keys)
        out.append(t.assign(in80=t.absz <= t.mult80, in95=t.absz <= t.mult95,
                            width80=2 * t.mult80 * t.sd / t.actual))
    return pd.concat(out).groupby("level")[["in80", "in95", "width80"]].mean().loc[LEVELS]


def main():
    choice = {}
    for measure in ["attendances", "emergency_admissions"]:
        b = standardised_errors(measure)
        tables, devs = {}, {}
        for name, keys in OPTIONS.items():
            cov = online_coverage(b, keys)
            tables[name] = cov
            devs[name] = float(((cov.in80 - 0.80).abs() + (cov.in95 - 0.95).abs()).mean())
        best = min(devs, key=devs.get)
        tbl = pd.concat(tables, axis=1)
        tbl.to_csv(RESULTS / f"interval_calibration_test_{measure}.csv")
        print(f"\n== {measure}: online coverage (origins 2024-07..2025-10). Mean |coverage - nominal|:")
        for k, v in devs.items():
            print(f"   {k:34s} {v:.3f}{'   <- chosen' if k == best else ''}")
        print(tbl.xs("in80", axis=1, level=1).round(3).to_string())

        keys = OPTIONS[best]
        mult = fit_multipliers(b, keys)  # live forecast: use every realised backtest error
        mult.to_csv(RESULTS / f"interval_multipliers_{measure}.csv", index=False)
        choice[measure] = {"method": best, "keys": keys, "mean_abs_coverage_gap": devs}
    (RESULTS / "interval_method.json").write_text(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
