import numpy as np
import pandas as pd
import pytest

from nhsae.models import COVID, covid_adjust, easter_month, stat_forecast


@pytest.mark.parametrize("year,month", [(2019, 4), (2024, 3), (2025, 4), (2027, 3), (2029, 4)])
def test_easter(year, month):
    assert easter_month(year) == month


def _seasonal(n=96, start="2018-04-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    rate = 1000 * 1.003 ** np.arange(n) * (1 + 0.06 * np.sin(2 * np.pi * idx.month / 12))
    return pd.DataFrame({"a": rate * idx.days_in_month, "b": 2 * rate * idx.days_in_month}, index=idx)


def test_covid_adjust_only_touches_covid_months():
    y = _seasonal()
    y.loc["2020-04":"2020-06"] *= 0.5
    adj = covid_adjust(y)
    outside = (y.index < COVID[0]) | (y.index > COVID[1])
    assert np.allclose(adj[outside], y[outside])
    assert (adj.loc["2020-04-01"] > y.loc["2020-04-01"]).all()


def test_seasonal_naive_uses_daily_rate():
    y = _seasonal()
    fc, _ = stat_forecast(y, 12, "snaive")
    last = y.iloc[-12:]
    expected = last.div(last.index.days_in_month, axis=0).to_numpy() * fc.index.days_in_month.to_numpy()[:, None]
    assert np.allclose(fc.to_numpy(), expected)


@pytest.mark.parametrize("model", ["ets", "sarima"])
def test_statistical_models_recover_a_clean_seasonal_series(model):
    y = _seasonal(n=108)
    train, test = y.iloc[:-12], y.iloc[-12:]
    fc, _ = stat_forecast(train, 12, model)
    mape = np.mean(np.abs(fc.to_numpy() - test.to_numpy()) / test.to_numpy())
    assert mape < 0.01
