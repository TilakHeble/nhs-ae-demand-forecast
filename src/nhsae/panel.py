"""Build a consistent trust → ICB → region → England panel for Type 1 (major) A&E.

Why this step matters: NHS trusts merge all the time. 21 Type 1 providers in the
2018-2026 files no longer exist under their old code. Forecasting the old codes
would mean modelling series that just stop, and the new trusts would look as if
their demand doubled overnight. We re-state history on today's structure
("as-is" restatement) by mapping each retired code to the trust it merged into,
then attach today's ICB and region from NHS England's provider-to-ICB mapping.
"""
from __future__ import annotations

import pandas as pd

from .config import HIERARCHY, PANEL, PROC
from .contracts import validate_panel
from .ingest import load_icb_mapping, load_monthly
from .reclass import detect as detect_reclass
from .reclass import restate as restate_reclass

OUT = PROC

# retired code -> successor code, with the first month the successor reported the combined activity.
# Every pair was checked by confirming the combined series is continuous across the merger month.
MERGERS = {
    "RGQ": ("RDE", "2018-07", "Ipswich + Colchester -> East Suffolk & North Essex"),
    "RR1": ("RRK", "2018-07", "Heart of England -> University Hospitals Birmingham"),
    "RJF": ("RTG", "2018-07", "Burton + Derby -> UH Derby & Burton"),
    "RLN": ("R0B", "2019-04", "City Hospitals Sunderland -> South Tyneside & Sunderland"),
    "RE9": ("R0B", "2019-04", "South Tyneside -> South Tyneside & Sunderland"),
    "RQ6": ("REM", "2019-10", "Royal Liverpool & Broadgreen -> Liverpool University Hospitals"),
    "RNL": ("RNN", "2019-10", "North Cumbria University Hospitals -> North Cumbria Integrated Care"),
    "RQ8": ("RAJ", "2020-04", "Mid Essex -> Mid & South Essex"),
    "RDD": ("RAJ", "2020-04", "Basildon & Thurrock -> Mid & South Essex"),
    "RA3": ("RA7", "2020-04", "Weston Area Health -> UH Bristol & Weston"),
    "RC1": ("RC9", "2020-04", "Bedford Hospital -> Bedfordshire Hospitals"),
    "RBA": ("RH5", "2020-04", "Taunton & Somerset -> Somerset FT"),
    "RD3": ("R0D", "2020-10", "Poole -> University Hospitals Dorset"),
    "RDZ": ("R0D", "2020-10", "Royal Bournemouth & Christchurch -> University Hospitals Dorset"),
    "RXH": ("RYR", "2021-04", "Brighton & Sussex -> University Hospitals Sussex"),
    "RW6": ("RM3", "2021-10", "Pennine Acute -> Northern Care Alliance (North Manchester site moved to MFT in Apr 2021)"),
    "RBZ": ("RH8", "2022-04", "Northern Devon -> Royal Devon University Healthcare"),
    "RA4": ("RH5", "2023-04", "Yeovil District -> Somerset FT"),
    "RVY": ("RBN", "2023-07", "Southport & Ormskirk -> Mersey & West Lancashire"),
    "RAP": ("RAL", "2025-01", "North Middlesex -> Royal Free London"),
    "RVJ": ("RA7", "2026-07", "North Bristol -> Bristol NHS FT"),
}

MEASURES = {
    "attendances": ["att_type1", "att_booked_type1"],
    "over_4h": ["over4h_type1", "over4h_booked_type1"],
    "emergency_admissions": ["emadm_type1"],
}


def build(min_months: int = 36) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_monthly()
    icb = load_icb_mapping()

    df = raw.copy()
    df["trust_code"] = df["org_code"].map(lambda c: MERGERS.get(c, (c,))[0])
    for name, cols in MEASURES.items():
        df[name] = df[[c for c in cols if c in df.columns]].sum(axis=1)

    t1 = (df[df["att_type1"] > 0]
          .groupby(["trust_code", "month"], as_index=False)[list(MEASURES) + ["att_type1", "att_other"]].sum())

    last = t1["month"].max()
    current = set(t1.loc[t1["month"] == last, "trust_code"])
    n_months = t1.groupby("trust_code")["month"].nunique()
    keep = sorted(c for c in current if n_months[c] >= min_months)
    t1 = t1[t1["trust_code"].isin(keep)]

    # complete grid (a trust with a zero month is a reporting gap, not true zero demand)
    grid = pd.MultiIndex.from_product([keep, sorted(t1["month"].unique())], names=["trust_code", "month"])
    t1 = t1.set_index(["trust_code", "month"]).reindex(grid).reset_index()

    names = (df[df["trust_code"].isin(keep) & (df["org_code"] == df["trust_code"])]
             .sort_values("month").groupby("trust_code")["org_name"].last())
    hier = (pd.DataFrame({"trust_code": keep})
            .merge(icb[["org_code", "icb_code", "icb_name", "region_name"]],
                   left_on="trust_code", right_on="org_code", how="left")
            .drop(columns="org_code"))
    hier["trust_name"] = hier["trust_code"].map(names).str.title().str.replace("Nhs", "NHS")
    missing = hier[hier["icb_code"].isna()]
    if len(missing):
        raise ValueError(f"Trusts with no ICB mapping: {missing.trust_code.tolist()}")

    panel = t1.merge(hier, on="trust_code")
    # 4-hour performance uses the figures as reported (breaches and attendances on the same basis)
    panel["attendances_reported"] = panel["attendances"]
    panel["pct_within_4h"] = 1 - panel["over_4h"] / panel["attendances_reported"]
    # demand series: restate Type 1 history onto each trust's current recording basis (see reclass.py)
    events = detect_reclass(panel[["trust_code", "month", "att_type1", "att_other"]])
    panel["attendances"] = restate_reclass(panel, events, "attendances")
    panel = panel.drop(columns=["att_type1", "att_other"])
    validate_panel(panel, hier)
    return panel, hier, events


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, hier, events = build()
    named = events.merge(hier[["trust_code", "trust_name"]], on="trust_code")
    named.to_csv(OUT / "type1_reclassifications.csv", index=False)
    panel.to_csv(PANEL, index=False)
    hier.to_csv(HIERARCHY, index=False)
    pd.DataFrame([{"retired_code": k, "successor_code": v[0], "first_combined_month": v[1], "note": v[2]}
                  for k, v in MERGERS.items()]).to_csv(OUT / "trust_mergers.csv", index=False)
    print(f"panel {panel.shape}, trusts={hier.trust_code.nunique()}, ICBs={hier.icb_code.nunique()}, "
          f"regions={hier.region_name.nunique()}, months={panel.month.nunique()}")
    print(f"Type 1 recording changes restated: {len(events)} events at {events.trust_code.nunique()} trusts")


if __name__ == "__main__":  # pragma: no cover
    main()
