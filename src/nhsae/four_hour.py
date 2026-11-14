"""4-hour performance risk: turn the demand forecast into a probability that each trust's Type 1 A&E
misses a 4-hour threshold in each coming month.

Why a separate model: 4-hour performance is a ratio, so it can't be reconciled up the hierarchy like
demand. It also depends on more than demand: on the trust's own recent performance (staffing, flow
and beds change slowly), on the time of year, and on how hard the whole system is running.

Model: one global LightGBM per quantile (5th…95th percentile), direct multi-horizon (h = 1..12).
Features use only information available at the forecast origin, plus the *demand forecast* for the
target month expressed as a ratio to the trust's last-12-month average (the "demand pressure").

  training : demand pressure uses actual demand (what happened)
  backtest : demand pressure uses the reconciled demand forecast made at that origin (what we knew)

From the quantiles we read off P(performance < threshold) for any threshold by interpolation.

Note: the national 4-hour objective (82% in 2026/27) covers all A&E types including urgent treatment
centres. This model covers Type 1 (major) departments only, where performance is lowest and where the
pressure sits, so thresholds here are set relative to Type 1 performance, not the all-types objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HIERARCHY, PANEL, RESULTS

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COVID_TARGETS = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"))
FEATURES = ["perf_last", "perf_last3", "perf_same_month_ly", "perf_last12", "perf_yoy_change",
            "nat_perf_last3", "demand_pressure", "demand_pressure_ly", "admit_rate_last3",
            "h", "month", "trust_size"]


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(PANEL, parse_dates=["month"])
    perf = p.pivot(index="month", columns="trust_code", values="pct_within_4h").asfreq("MS")
    att = p.pivot(index="month", columns="trust_code", values="attendances").asfreq("MS")
    adm = p.pivot(index="month", columns="trust_code", values="emergency_admissions").asfreq("MS")
    # national Type 1 performance from reported figures (breaches and attendances on the same recording basis)
    tot = p.groupby("month")[["over_4h", "attendances_reported"]].sum().asfreq("MS")
    perf.attrs["national"] = (1 - tot.over_4h / tot.attendances_reported).to_numpy()
    return perf, att, adm


def make_rows(perf, att, adm, o: int, h: int, demand_target: np.ndarray) -> pd.DataFrame:
    """Feature rows for every trust: origin index o, horizon h, demand for the target month supplied."""
    months = perf.index
    tgt = months[o] + pd.DateOffset(months=h)
    P, A, M = perf.to_numpy(), att.to_numpy(), adm.to_numpy()
    nat = perf.attrs["national"]                         # national Type 1 performance by month
    d = months.days_in_month.to_numpy(float)[:, None]
    rate = A / d                                          # daily attendances
    base = rate[o - 11:o + 1].mean(0)
    tgt_rate = demand_target / tgt.days_in_month
    ly = o + h - 12
    return pd.DataFrame({
        "trust_code": perf.columns, "origin": months[o], "target_month": tgt,
        "perf_last": P[o], "perf_last3": P[o - 2:o + 1].mean(0),
        "perf_same_month_ly": P[ly], "perf_last12": P[o - 11:o + 1].mean(0),
        "perf_yoy_change": P[o - 2:o + 1].mean(0) - P[o - 14:o - 11].mean(0),
        "nat_perf_last3": np.full(P.shape[1], nat[o - 2:o + 1].mean()),
        "demand_pressure": tgt_rate / base,
        # same definition one year earlier: last year's target-month demand vs the 12 months before last year's origin
        "demand_pressure_ly": rate[ly] / rate[o - 23:o - 11].mean(0) if o >= 23 else np.full(P.shape[1], np.nan),
        "admit_rate_last3": M[o - 2:o + 1].sum(0) / A[o - 2:o + 1].sum(0),
        "h": h, "month": tgt.month, "trust_size": np.log(base),
    })


def training_set(perf, att, adm, last_origin: int, H: int = 12) -> pd.DataFrame:
    rows = []
    months = perf.index
    for o in range(14, last_origin):
        for h in range(1, H + 1):
            t = o + h
            if t > last_origin:
                break
            if COVID_TARGETS[0] <= months[t] <= COVID_TARGETS[1] or COVID_TARGETS[0] <= months[o] <= COVID_TARGETS[1]:
                continue
            r = make_rows(perf, att, adm, o, h, att.iloc[t].to_numpy())
            r["y"] = perf.iloc[t].to_numpy()
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def fit_quantiles(train: pd.DataFrame) -> dict:
    import lightgbm as lgb
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, learning_rate=0.03,
                              num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(train[FEATURES], train["y"])
        models[q] = m
    return models


def predict_quantiles(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({f"q{int(q * 100):02d}": models[q].predict(X[FEATURES]) for q in QUANTILES}, index=X.index)
    out[:] = np.sort(out.to_numpy(), axis=1)  # enforce non-crossing quantiles
    return out.clip(0, 1)


def prob_below(qrow: np.ndarray, thr: float) -> float:
    """Predictive CDF at `thr`: linear interpolation between the quantiles, and Gaussian tails beyond
    the 5th / 95th percentile (sd implied by the 5-95 spread) so that probabilities keep ranking trusts
    sensibly even when the threshold is far outside the forecast range."""
    from scipy.stats import norm
    qs = np.array(QUANTILES)
    sd = max((qrow[-1] - qrow[0]) / (2 * norm.ppf(0.95)), 1e-4)
    if thr <= qrow[0]:
        return float(norm.cdf(norm.ppf(qs[0]) + (thr - qrow[0]) / sd))
    if thr >= qrow[-1]:
        return float(norm.cdf(norm.ppf(qs[-1]) + (thr - qrow[-1]) / sd))
    return float(np.interp(thr, qrow, qs))


QCOLS = [f"q{int(a * 100):02d}" for a in QUANTILES]


def conformal_offsets(cal: pd.DataFrame) -> np.ndarray:
    """Conformalised quantile regression, one offset per quantile level: shift each predicted quantile
    by the empirical a-quantile of (actual - predicted a-quantile) seen on already-realised forecasts,
    so each quantile is marginally calibrated."""
    return np.array([np.quantile(cal.y - cal[c], a) for c, a in zip(QCOLS, QUANTILES)])


def shift(df: pd.DataFrame, off: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    vals = np.sort(out[QCOLS].to_numpy() + off, axis=1)
    out[QCOLS] = np.clip(vals, 0, 1)
    return out


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


# ------------------------------------------------------------------ backtest
def backtest() -> pd.DataFrame:
    perf, att, adm = load_panel()
    fc = pd.read_parquet(RESULTS / "backtest_attendances_post.parquet",
                         columns=["origin", "model", "recon", "h", "series", "level", "forecast"])
    fc = fc[(fc.model == "combo") & (fc.recon == "mint_shrink") & (fc.level == "Trust")]
    fc["trust_code"] = fc.series.str[2:]
    months = perf.index
    out = []
    for origin in sorted(fc.origin.unique()):
        o = months.get_loc(origin)
        models = fit_quantiles(training_set(perf, att, adm, o))
        f_o = fc[fc.origin == origin]
        for h in range(1, 13):
            if o + h >= len(months):
                break
            dem = f_o[f_o.h == h].set_index("trust_code").forecast.reindex(perf.columns).to_numpy()
            X = make_rows(perf, att, adm, o, h, dem)
            q = predict_quantiles(models, X)
            X = pd.concat([X, q], axis=1)
            X["y"] = perf.iloc[o + h].to_numpy()
            out.append(X)
    raw = pd.concat(out, ignore_index=True)
    # online conformal step: each origin is calibrated only on earlier forecasts whose target month has happened
    cal_rows = []
    for origin in sorted(raw.origin.unique()):
        g = raw[raw.origin == origin]
        seen = raw[(raw.origin < origin) & (raw.target_month <= origin)]
        cal_rows.append(shift(g, conformal_offsets(seen)) if len(seen) > 500 else g.assign(_uncal=True))
    bt = pd.concat(cal_rows, ignore_index=True)
    bt = bt[bt.get("_uncal", pd.Series(False, index=bt.index)).ne(True)].drop(columns=["_uncal"], errors="ignore")
    raw.to_parquet(RESULTS / "four_hour_backtest_raw.parquet", index=False)
    bt.to_parquet(RESULTS / "four_hour_backtest.parquet", index=False)
    return raw, bt


def evaluate(bt: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    y = bt.y.to_numpy()
    rows = {
        "LightGBM quantile model (median)": bt.q50,
        "Naive: last month": bt.perf_last,
        "Naive: same month last year": bt.perf_same_month_ly,
        "Naive: last 3 months mean": bt.perf_last3,
    }
    res = pd.DataFrame({k: {"MAE (pct points)": 100 * np.mean(np.abs(y - v))} for k, v in rows.items()}).T
    res["Pinball loss (avg over quantiles, ×100)"] = np.nan
    res.loc["LightGBM quantile model (median)", "Pinball loss (avg over quantiles, ×100)"] = 100 * np.mean(
        [pinball(y, bt[f"q{int(a * 100):02d}"], a) for a in QUANTILES])
    cov80 = np.mean((y >= bt.q10) & (y <= bt.q90))
    cov90 = np.mean((y >= bt.q05) & (y <= bt.q95))

    # calibration of event probabilities: miss 60% (≈ national Type 1 level) and "worse than last year"
    cal = []
    for name, thr in [("below 60%", np.full(len(bt), 0.60)), ("worse than same month last year", bt.perf_same_month_ly)]:
        qs = bt[[f"q{int(a * 100):02d}" for a in QUANTILES]].to_numpy()
        p = np.array([prob_below(qs[i], thr.iloc[i] if hasattr(thr, "iloc") else thr[i]) for i in range(len(bt))])
        ev = (y < (thr.to_numpy() if hasattr(thr, "to_numpy") else thr)).astype(float)
        base = np.full(len(ev), ev.mean())
        bins = pd.cut(p, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
        tab = pd.DataFrame({"p": p, "ev": ev, "bin": bins}).groupby("bin", observed=True).agg(
            predicted=("p", "mean"), observed=("ev", "mean"), n=("ev", "size"))
        tab["event"] = name
        cal.append(tab.reset_index())
        res.loc[f"Brier score: {name}", "MAE (pct points)"] = np.nan
        res.loc[f"Brier score: {name}", "Brier"] = np.mean((p - ev) ** 2)
        res.loc[f"Brier score: {name}", "Brier (climatology)"] = np.mean((base - ev) ** 2)
    pd.concat(cal).to_csv(RESULTS / "four_hour_calibration.csv", index=False)
    res.loc["Interval coverage (conformal)", "80% (q10-q90)"] = cov80
    res.loc["Interval coverage (conformal)", "90% (q05-q95)"] = cov90
    if raw is not None:
        r = raw[raw.origin.isin(bt.origin.unique())]
        res.loc["Interval coverage (raw quantiles)", "80% (q10-q90)"] = np.mean((r.y >= r.q10) & (r.y <= r.q90))
        res.loc["Interval coverage (raw quantiles)", "90% (q05-q95)"] = np.mean((r.y >= r.q05) & (r.y <= r.q95))
    res.to_csv(RESULTS / "four_hour_backtest_summary.csv")
    return res


# ------------------------------------------------------------------ live forecast
def live() -> pd.DataFrame:
    perf, att, adm = load_panel()
    o = len(perf.index) - 1
    models = fit_quantiles(training_set(perf, att, adm, o))
    fc = pd.read_csv(RESULTS / "forecast_attendances.csv", parse_dates=["month"])
    fc = fc[fc.level == "Trust"]
    fc["trust_code"] = fc.series.str[2:]
    out = []
    for h in range(1, 13):
        dem = fc[fc.h == h].set_index("trust_code").forecast.reindex(perf.columns).to_numpy()
        X = make_rows(perf, att, adm, o, h, dem)
        out.append(pd.concat([X, predict_quantiles(models, X)], axis=1))
    raw_bt = pd.read_parquet(RESULTS / "four_hour_backtest_raw.parquet")
    live = shift(pd.concat(out, ignore_index=True), conformal_offsets(raw_bt))
    qs = live[[f"q{int(a * 100):02d}" for a in QUANTILES]].to_numpy()
    live["p_below_60"] = [prob_below(r, 0.60) for r in qs]
    live["p_worse_than_ly"] = [prob_below(qs[i], live.perf_same_month_ly.iloc[i]) for i in range(len(live))]
    hier = pd.read_csv(HIERARCHY)
    live = live.merge(hier[["trust_code", "trust_name", "icb_name", "region_name"]], on="trust_code")
    live.to_csv(RESULTS / "four_hour_forecast.csv", index=False)

    imp = pd.Series(models[0.5].booster_.feature_importance("gain"), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_csv(RESULTS / "four_hour_feature_importance.csv")
    return live


def main():
    raw, bt = backtest()
    print(evaluate(bt, raw).round(4).to_string())
    lv = live()
    w = lv[lv.target_month.dt.month.isin([12, 1, 2])]
    w = w[w.target_month <= w.target_month.min() + pd.DateOffset(months=2)]
    top = (w.groupby(["trust_name", "region_name"]).agg(median_perf=("q50", "mean"), p_below_60=("p_below_60", "mean"))
           .sort_values("p_below_60", ascending=False))
    print("\nWinter (Dec–Feb) highest-risk trusts:\n", top.head(10).round(3).to_string())
    imp = pd.read_csv(RESULTS / "four_hour_feature_importance.csv")
    print("\nfeature importance:\n", imp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
