"""Load NHS England Monthly A&E Attendances & Emergency Admissions files into one tidy table.

The monthly CSVs are published one per month. Column names drift over the years
("A&E attendances Type 1" vs "Number of A&E attendances Type 1", stray "Unnamed"
columns, a "TOTAL" row, trailing spaces in region names). This module normalises
all of that into a single provider-month table with short canonical column names.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# canonical name -> regex matched against a normalised (lower-case, collapsed) header
COLUMN_PATTERNS = {
    "att_type1": r"^(number of )?a&e attendances type 1$",
    "att_type2": r"^(number of )?a&e attendances type 2$",
    "att_other": r"^(number of )?a&e attendances other( a&e)? department$",
    "att_booked_type1": r"^a&e attendances booked appointments type 1$",
    "att_booked_type2": r"^a&e attendances booked appointments type 2$",
    "att_booked_other": r"^a&e attendances booked appointments other department$",
    "over4h_type1": r"^(number of )?attendances over 4hrs type 1$",
    "over4h_type2": r"^(number of )?attendances over 4hrs type 2$",
    "over4h_other": r"^(number of )?attendances over 4hrs other( a&e)? department$",
    "over4h_booked_type1": r"^attendances over 4hrs booked appointments type 1$",
    "over4h_booked_type2": r"^attendances over 4hrs booked appointments type 2$",
    "over4h_booked_other": r"^attendances over 4hrs booked appointments other department$",
    "dta_4_12h": r"^(number of )?patients who have waited 4-12 hs? from dta to admission\.?$",
    "dta_12h_plus": r"^(number of )?patients who have waited 12\+ hrs from dta to admission\.?$",
    "emadm_type1": r"^(number of )?emergency admissions via (a&e - )?type 1$",
    "emadm_type2": r"^(number of )?emergency admissions via (a&e - )?type 2$",
    "emadm_other_ae": r"^(number of )?emergency admissions via (a&e - )?other a&e department$",
    "emadm_other": r"^(number of )?other emergency admissions$",
}
ID_COLS = {"period": r"^period$", "org_code": r"^org code$", "region_raw": r"^parent org$", "org_name": r"^org name$"}


def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        n = _norm(col)
        for canon, pat in {**ID_COLS, **COLUMN_PATTERNS}.items():
            if re.match(pat, n):
                rename[col] = canon
                break
    out = df.rename(columns=rename)
    keep = [c for c in list(ID_COLS) + list(COLUMN_PATTERNS) if c in out.columns]
    return out[keep]


def _split_bundle(path: Path) -> list[tuple[str, str]]:
    """The browser bundle is many CSVs joined with '### FILE: <label> | <url>' headers."""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^### FILE: (.*)\n", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]


def load_monthly(raw_dir: Path = RAW) -> pd.DataFrame:
    frames = []
    bundle = raw_dir / "nhs_ae_monthly_bundle.txt"
    if bundle.exists():
        for label, body in _split_bundle(bundle):
            d = _standardise(pd.read_csv(io.StringIO(body)))
            d["source"] = label.split(" | ")[-1]
            frames.append(d)
    for f in sorted(raw_dir.glob("*.csv")):
        d = _standardise(pd.read_csv(f))
        d["source"] = f.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # drop totals / blank rows
    df = df[df["org_code"].notna() & ~df["org_code"].astype(str).str.upper().isin(["-", "TOTAL"])]
    per = df["period"].astype(str).str.upper().str.replace("MSITAE-", "", regex=False).str.strip()
    df = df[~per.str.contains("TOTAL")]
    per = per[df.index]
    df["month"] = pd.to_datetime(per.str.title(), format="%B-%Y")

    num_cols = [c for c in COLUMN_PATTERNS if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip().replace({"-": None, "nan": None}),
                              errors="coerce").fillna(0)
    df["org_code"] = df["org_code"].astype(str).str.strip().str.upper()
    df["org_name"] = df["org_name"].astype(str).str.strip()
    df["region_raw"] = df["region_raw"].astype(str).str.strip().str.upper()

    # a month can appear twice if a revised file was published; keep the last-published copy
    df = df.drop_duplicates(subset=["month", "org_code"], keep="first")
    return df.drop(columns=["period"]).sort_values(["month", "org_code"]).reset_index(drop=True)


def load_icb_mapping(raw_dir: Path = RAW) -> pd.DataFrame:
    m = pd.read_excel(raw_dir / "nhs_ae_System-Mapping-Oct-26.xls", header=7).iloc[:, 1:6]
    m.columns = ["org_code", "org_name_map", "icb_code", "icb_name", "region_name"]
    m = m.dropna(subset=["org_code"])
    m["org_code"] = m["org_code"].astype(str).str.strip().str.upper()
    m["region_name"] = m["region_name"].str.replace(" Commissioning Region", "", regex=False).str.strip()
    m["icb_name"] = (m["icb_name"].str.replace("Integrated Care Board", "ICB", regex=False)
                     .str.replace(r"^NHS ", "", regex=True).str.strip())
    return m.drop_duplicates("org_code")


if __name__ == "__main__":
    d = load_monthly()
    print(d.shape, d.month.min().date(), d.month.max().date(), d.month.nunique())
