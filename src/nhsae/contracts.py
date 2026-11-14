"""Data contracts: checks the processed panel must pass before anything is modelled.

They run every time the panel is built (and in CI), so a changed NHS file format, a missed merger
or a broken mapping fails loudly instead of quietly producing wrong forecasts.
"""
from __future__ import annotations

import pandas as pd

EXPECTED_REGIONS = 7


class ContractError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def validate_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """The harmonised raw provider-month table."""
    months = pd.date_range(raw.month.min(), raw.month.max(), freq="MS")
    _check(raw.month.nunique() == len(months), f"missing months: {sorted(set(months) - set(raw.month))[:5]}")
    _check(not raw.duplicated(["month", "org_code"]).any(), "duplicate provider-months")
    num = [c for c in raw.columns if c.startswith(("att_", "over4h_", "emadm_", "dta_"))]
    _check(len(num) >= 10, f"only {len(num)} measure columns recognised: column mapping broken?")
    _check((raw[num] >= 0).all().all(), "negative counts")
    return raw


def validate_panel(panel: pd.DataFrame, hier: pd.DataFrame) -> pd.DataFrame:
    """The trust-month panel used for modelling."""
    _check(panel[["attendances", "over_4h", "emergency_admissions"]].notna().all().all(), "gaps in the panel")
    _check((panel.attendances > 0).all(), "a trust-month has zero Type 1 attendances")
    _check((panel.over_4h <= panel.attendances).all(), "4-hour breaches exceed attendances")
    _check(panel.pct_within_4h.between(0, 1).all(), "4-hour performance outside [0, 1]")
    n_months = panel.month.nunique()
    _check((panel.groupby("trust_code").month.nunique() == n_months).all(), "unbalanced panel")
    _check(hier.trust_code.is_unique, "trust appears twice in the hierarchy")
    _check(hier.icb_code.notna().all(), "trust without an ICB")
    _check(hier.region_name.nunique() == EXPECTED_REGIONS, f"expected {EXPECTED_REGIONS} regions")
    _check((hier.groupby("icb_code").region_name.nunique() == 1).all(), "an ICB spans two regions")
    # merger restatement check: no trust's daily rate should jump >60% month on month after mid-2021
    w = panel.pivot(index="month", columns="trust_code", values="attendances").loc["2021-07":]
    rate = w.div(w.index.days_in_month, axis=0)
    jump = (rate / rate.shift(1)).iloc[1:]
    bad = jump.columns[((jump > 1.6) | (jump < 1 / 1.6)).any()]
    _check(len(bad) == 0, f"step changes suggest an unmapped merger: {list(bad)}")
    return panel
