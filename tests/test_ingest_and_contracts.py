import io

import pandas as pd
import pytest

from nhsae.config import MONTHLY_BUNDLE
from nhsae.contracts import ContractError, validate_panel
from nhsae.ingest import _split_bundle, _standardise, load_monthly

pytestmark = pytest.mark.skipif(not MONTHLY_BUNDLE.exists(), reason="raw data absent")


def test_header_variants_map_to_one_schema():
    old = pd.read_csv(io.StringIO("Period,Org Code,Parent Org,Org name,Number of A&E attendances Type 1,"
                                  "Number of attendances over 4hrs Type 1,Unnamed: 22\nMSitAE-JUNE-2018,RXX,R,T,10,2,\n"))
    new = pd.read_csv(io.StringIO("Period,Org Code,Parent Org,Org name,A&E attendances Type 1,"
                                  "Attendances over 4hrs Type 1\nMSitAE-JUNE-2026,RXX,R,T,10,2\n"))
    assert list(_standardise(old).columns) == list(_standardise(new).columns)
    assert {"att_type1", "over4h_type1"} <= set(_standardise(old).columns)


def test_bundle_has_every_month():
    parts = _split_bundle(MONTHLY_BUNDLE)
    assert len(parts) >= 100
    raw = load_monthly()  # runs the raw data contract
    months = pd.date_range("2018-04-01", raw.month.max(), freq="MS")
    assert set(months) == set(raw.month)
    assert not raw.duplicated(["month", "org_code"]).any()


def test_processed_panel_passes_contract(panel, hier):
    validate_panel(panel, hier)
    assert hier.trust_code.nunique() == 120
    assert hier.icb_code.nunique() == 36
    assert panel.month.nunique() == len(pd.date_range(panel.month.min(), panel.month.max(), freq="MS"))


def test_contract_catches_unmapped_step_change(panel, hier):
    bad = panel.copy()
    mask = (bad.trust_code == bad.trust_code.iloc[0]) & (bad.month >= "2024-01-01")
    bad.loc[mask, "attendances"] *= 2.0  # looks like a merger nobody mapped
    with pytest.raises(ContractError, match="step changes"):
        validate_panel(bad, hier)


def test_contract_catches_impossible_counts(panel, hier):
    bad = panel.copy()
    bad.loc[bad.index[0], "over_4h"] = bad.loc[bad.index[0], "attendances"] + 1
    with pytest.raises(ContractError):
        validate_panel(bad, hier)


def test_mergers_and_restatement_keep_england_total(panel):
    # restating recording changes moves activity between trusts' Type 1 and "other"; England moves little
    eng = panel.groupby("month")[["attendances", "attendances_reported"]].sum()
    assert (eng.attendances / eng.attendances_reported - 1).abs().max() < 0.02
