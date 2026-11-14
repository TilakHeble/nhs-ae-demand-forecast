"""Base forecasting models for every series in the hierarchy.

All models forecast the *daily rate* (attendances / days in month) on the log scale and convert back
to monthly totals. That removes the February effect and makes seasonal patterns multiplicative.

COVID: April 2020 - June 2021 is not normal demand. Each model sees a version of history where
those months are replaced by a counterfactual (same month in 2019 x the series' 2021-22 recovery
ratio), plus the option to train only on post-COVID data (see backtest `--covid` flag).

Models
  snaive  : same month last year (benchmark every model must beat)
  ets     : Holt-Winters, damped additive trend + additive seasonality on log daily rate
  sarima  : airline model ARIMA(0,1,1)(0,1,1)12 on log daily rate
  lgbm    : one global LightGBM across all 164 series, direct multi-horizon (h = 1..12)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

COVID = (pd.Timestamp("2020-04-01"), pd.Timestamp("2021-06-01"))


def easter_month(year: int) -> int:
    """Anonymous Gregorian algorithm -> month of Easter Sunday (3 or 4)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return (h + l - 7 * m + 114) // 31


def days(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.days_in_month.to_numpy(float)


def covid_adjust(y: pd.DataFrame, mode: str = "impute") -> pd.DataFrame:
    """y: monthly totals (index month, columns series). Returns adjusted copy.

    impute : replace COVID months with 2019 same-month daily rate x recovery ratio (Jul21-Jun22 vs Jul18-Jun19)
    none   : leave as is
    """
    if mode == "none":
        return y
    out = y.copy()
    rate = y.div(days(y.index), axis=0)
    pre = rate.loc["2018-07":"2019-06"].mean()
    post = rate.loc["2021-07":"2022-06"].mean()
    if post.isna().all() or pre.isna().all():
        return out
    ratio = (post / pre).clip(0.5, 2.0)
    for mth in pd.date_range(*COVID, freq="MS"):
        if mth not in out.index:
            continue
        ref = mth.replace(year=2019)
        # linearly blend from the 2019 level to the post-COVID level across the window
        w = (mth - COVID[0]).days / (COVID[1] - COVID[0]).days
        adj = rate.loc[ref] * (1 + w * (ratio - 1))
        out.loc[mth] = adj * mth.days_in_month
    return out


# ---------------- statistical models (one fit per series) ----------------

def _fit_ets(z: np.ndarray, H: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    mod = ExponentialSmoothing(z, trend="add", damped_trend=True, seasonal="add",
                               seasonal_periods=12, initialization_method="estimated")
    fit = mod.fit(optimized=True)
    return fit.forecast(H), fit.fittedvalues


def _fit_sarima(z: np.ndarray, H: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(z, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    fitted = fit.fittedvalues.copy()
    fitted[:13] = np.nan  # differencing burn-in
    return fit.forecast(H), fitted


def stat_forecast(y: pd.DataFrame, H: int, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """y: monthly totals (T x n). Returns (forecasts H x n, in-sample one-step residuals T x n), monthly totals."""
    idx_f = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")
    d_in, d_out = days(y.index), days(idx_f)
    fc = np.zeros((H, y.shape[1]))
    res = np.full(y.shape, np.nan)
    for j, col in enumerate(y.columns):
        s = y[col].to_numpy(float)
        if model == "snaive":
            last = s[-12:] / d_in[-12:]
            fc[:, j] = np.resize(last, H) * d_out
            fitted = np.r_[np.full(12, np.nan), s[:-12] / d_in[:-12] * d_in[12:]]
            res[:, j] = s - fitted
            continue
        z = np.log(np.maximum(s, 1) / d_in)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, fitted = (_fit_ets if model == "ets" else _fit_sarima)(z, H)
        fc[:, j] = np.exp(f) * d_out
        res[:, j] = s - np.exp(fitted) * d_in
    return pd.DataFrame(fc, index=idx_f, columns=y.columns), pd.DataFrame(res, index=y.index, columns=y.columns)


# ---------------- global LightGBM (direct multi-horizon) ----------------

def _features(rate: np.ndarray, months: pd.DatetimeIndex, o: int, h: int, level_code: np.ndarray) -> np.ndarray:
    """Features for target month o+h made only from information available at origin o (array rows)."""
    tgt = months[o] + pd.DateOffset(months=h)
    scale = rate[o - 11:o + 1].mean(axis=0)               # last 12 months mean daily rate
    lr = lambda a: np.log(np.maximum(a, 1e-9) / scale)  # noqa: E731
    same_month_ly = rate[o + h - 12]                       # needs h <= 12
    last3 = rate[o - 2:o + 1].mean(axis=0)
    last3_ly = rate[o - 14:o - 11].mean(axis=0)
    eas = easter_month(tgt.year)
    n = rate.shape[1]
    return np.column_stack([
        lr(same_month_ly), lr(last3), lr(last3_ly), lr(last3) - lr(last3_ly),
        lr(rate[o]), lr(rate[o - 1]),
        np.full(n, h), np.full(n, tgt.month), np.full(n, float(tgt.month == eas)),
        np.full(n, tgt.days_in_month), level_code,
    ]), scale


def lgbm_forecast(y: pd.DataFrame, H: int, levels: np.ndarray, seed: int = 0):
    import lightgbm as lgb

    lvl_map = {"England": 0, "Region": 1, "ICB": 2, "Trust": 3}
    level_code = np.array([lvl_map[v] for v in levels], float)
    months = y.index
    rate = y.to_numpy(float) / days(months)[:, None]
    T = len(months)

    X, Y = [], []
    for o in range(14, T - 1):
        for h in range(1, min(H, 12) + 1):
            if o + h >= T:
                break
            f, scale = _features(rate, months, o, h, level_code)
            X.append(f)
            Y.append(np.log(rate[o + h] / scale))
    X, Y = np.vstack(X), np.concatenate(Y)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
    model.fit(X, Y, categorical_feature=[10])

    idx_f = pd.date_range(months[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")
    fc = np.zeros((H, y.shape[1]))
    for h in range(1, H + 1):
        f, scale = _features(rate, months, T - 1, h, level_code)
        fc[h - 1] = np.exp(model.predict(f)) * scale * idx_f[h - 1].days_in_month

    # one-step in-sample residuals for MinT
    res = np.full(y.shape, np.nan)
    for o in range(14, T - 1):
        f, scale = _features(rate, months, o, 1, level_code)
        res[o + 1] = y.iloc[o + 1].to_numpy() - np.exp(model.predict(f)) * scale * months[o + 1].days_in_month
    return (pd.DataFrame(fc, index=idx_f, columns=y.columns),
            pd.DataFrame(res, index=months, columns=y.columns), model)


def base_forecast(y: pd.DataFrame, H: int, model: str, levels: np.ndarray):
    if model == "lgbm":
        fc, res, _ = lgbm_forecast(y, H, levels)
        return fc, res
    return stat_forecast(y, H, model)
