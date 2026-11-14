"""Checks on the committed outputs: coherence, sane intervals, valid probabilities."""
import numpy as np
import pandas as pd
import pytest

from nhsae.config import RESULTS
from nhsae.hierarchy import Hierarchy

MEASURES = ["attendances", "emergency_admissions"]


@pytest.mark.parametrize("measure", MEASURES)
def test_live_forecasts_are_coherent_at_every_level(measure, hier):
    fc = pd.read_csv(RESULTS / f"forecast_{measure}.csv")
    H = Hierarchy(hier)
    wide = fc.pivot(index="month", columns="series", values="forecast")[H.ids].to_numpy()
    bottom = wide[:, -H.m:]
    assert np.allclose(wide, bottom @ H.S.T, rtol=1e-8)


@pytest.mark.parametrize("measure", MEASURES)
def test_intervals_are_ordered(measure):
    fc = pd.read_csv(RESULTS / f"forecast_{measure}.csv")
    assert (fc.lo95 <= fc.lo80).all() and (fc.lo80 <= fc.forecast).all()
    assert (fc.forecast <= fc.hi80).all() and (fc.hi80 <= fc.hi95).all()
    assert (fc.lo95 >= 0).all()


def test_four_hour_quantiles_and_probabilities():
    lv = pd.read_csv(RESULTS / "four_hour_forecast.csv")
    q = lv[["q05", "q10", "q25", "q50", "q75", "q90", "q95"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert ((q >= 0) & (q <= 1)).all()
    assert lv.p_below_60.between(0, 1).all() and lv.p_worse_than_ly.between(0, 1).all()
    assert lv.trust_code.nunique() == 120


def test_best_model_beats_seasonal_naive_everywhere():
    for measure in MEASURES:
        s = pd.read_csv(RESULTS / f"backtest_summary_{measure}.csv")
        f = pd.read_csv(RESULTS / f"forecast_{measure}.csv")
        model, recon, covid = f.model.iloc[0], f.recon.iloc[0], f.covid.iloc[0]
        best = s[(s.model == model) & (s.recon == recon) & (s.covid == covid)].set_index("level").MASE
        naive = s[(s.model == "snaive") & (s.recon == "base") & (s.covid == covid)].set_index("level").MASE
        assert (best < naive).all(), measure
