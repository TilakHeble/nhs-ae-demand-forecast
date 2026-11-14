import numpy as np
import pandas as pd

from nhsae.reclass import detect, restate


def _series(t1, other, start="2022-01-01"):
    m = pd.date_range(start, periods=len(t1), freq="MS")
    d = m.days_in_month.to_numpy()
    return pd.DataFrame({"trust_code": "TST", "month": m, "att_type1": np.array(t1) * d, "att_other": np.array(other) * d})


def test_detects_streaming_to_utc_and_restates_continuously():
    n, k = 36, 20
    season = 1 + 0.03 * np.sin(np.arange(n) * 2 * np.pi / 12)
    total = 500 * season
    t1 = np.where(np.arange(n) < k, total, 0.6 * total)        # 40% of patients streamed away from month k
    other = total - t1
    df = _series(t1, other)
    ev = detect(df)
    assert len(ev) == 1
    assert ev.month.iloc[0] == df.month.iloc[k]
    assert abs(ev.factor.iloc[0] - 0.6) < 0.02
    s = df.assign(attendances=df.att_type1)
    restated = restate(s, ev) / s.month.dt.days_in_month
    jump = restated.iloc[k] / restated.iloc[k - 1]
    assert abs(jump - 1) < 0.05


def test_no_false_positive_on_genuine_demand_growth():
    n = 36
    t1 = 400 * 1.01 ** np.arange(n) * (1 + 0.05 * np.sin(np.arange(n) * 2 * np.pi / 12))
    other = 100 * 1.01 ** np.arange(n)
    assert detect(_series(t1, other)).empty
