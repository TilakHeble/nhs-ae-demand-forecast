"""Detect and restate Type 1 ↔ Type 3 recording changes.

Many trusts open a co-located urgent treatment centre (UTC) and start streaming minor cases there.
Overnight, thousands of attendances a month move from "Type 1" to "Other A&E department" in the
returns, or the reverse when a UTC is folded back into the main department. Total activity barely
moves, but the Type 1 series shows a step that a forecasting model reads as a collapse or a surge in
demand.

Detection: at each month m, compare 3-month mean daily rates before and after. It is flagged as a
reclassification when
  * Type 1 changes by more than 15%, and
  * "Other" moves in the opposite direction, and
  * the two changes largely cancel (|ΔType1 + ΔOther| < 50% of |ΔType1|).
Flagged months within 4 months of each other are one event; the month with the largest change is used.
Windows that touch the COVID period are skipped.

Restatement: history before an event is rescaled by s_post / s_pre, where s is the Type 1 share of
(Type 1 + Other) in the 3 months after / before. That puts all history on the trust's *current*
recording basis, the same "as-is" principle used for mergers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVID_WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-09-01"))
MIN_T1_CHANGE = 0.15
MAX_NET_RATIO = 0.5
WINDOW = 3


def detect(trust_month: pd.DataFrame) -> pd.DataFrame:
    """trust_month: columns trust_code, month, att_type1, att_other (successor codes already applied)."""
    events = []
    for trust, d in trust_month.groupby("trust_code"):
        d = d.set_index("month").sort_index()[["att_type1", "att_other"]]
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="MS")).fillna(0)
        rate = d.div(d.index.days_in_month, axis=0)
        flagged = []
        for i in range(WINDOW, len(rate) - WINDOW + 1):
            m = rate.index[i]
            if COVID_WINDOW[0] <= m <= COVID_WINDOW[1]:
                continue
            pre, post = rate.iloc[i - WINDOW:i].mean(), rate.iloc[i:i + WINDOW].mean()
            if pre.att_type1 <= 0:
                continue
            d1, d3 = post.att_type1 - pre.att_type1, post.att_other - pre.att_other
            if (abs(d1) > MIN_T1_CHANGE * pre.att_type1 and np.sign(d1) != np.sign(d3)
                    and abs(d1 + d3) < MAX_NET_RATIO * abs(d1)):
                s_pre = pre.att_type1 / (pre.att_type1 + pre.att_other)
                s_post = post.att_type1 / (post.att_type1 + post.att_other)
                flagged.append({"trust_code": trust, "month": m, "t1_change": d1 / pre.att_type1,
                                "t1_daily_pre": pre.att_type1, "t1_daily_post": post.att_type1,
                                "other_daily_pre": pre.att_other, "other_daily_post": post.att_other,
                                "share_pre": s_pre, "share_post": s_post, "factor": s_post / s_pre})
        if not flagged:
            continue
        f = pd.DataFrame(flagged)
        cluster = (f.month.diff().dt.days.fillna(999) > 4 * 31).cumsum()
        for _, g in f.groupby(cluster):
            events.append(g.loc[g.t1_change.abs().idxmax()])
    cols = ["trust_code", "month", "t1_change", "t1_daily_pre", "t1_daily_post", "other_daily_pre",
            "other_daily_post", "share_pre", "share_post", "factor"]
    return pd.DataFrame(events, columns=cols).reset_index(drop=True)


def restate(series: pd.DataFrame, events: pd.DataFrame, value: str = "attendances") -> pd.Series:
    """Rescale `value` for months before each event by that event's factor (cumulative, latest first)."""
    out = series[value].astype(float).copy()
    for _, e in events.sort_values("month", ascending=False).iterrows():
        mask = (series.trust_code == e.trust_code) & (series.month < e.month)
        out[mask] *= e.factor
    return out
