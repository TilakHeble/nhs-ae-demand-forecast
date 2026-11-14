import pytest

from nhsae.config import RESULTS, ROOT

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (RESULTS / "forecast_attendances.csv").exists(), reason="results absent")


def test_dashboard_runs_and_responds():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception
    england_total = next(m.value for m in at.metric if m.label.startswith("Next 12 months"))
    at.sidebar.selectbox[0].set_value("London").run()          # drill into a region
    assert not at.exception
    assert next(m.value for m in at.metric if m.label.startswith("Next 12 months")) != england_total
    below = next(m for m in at.metric if m.label.startswith("Trusts more likely than not below"))
    at.slider[0].set_value(80).run()                          # raise the 4-hour threshold
    assert not at.exception
    above = next(m for m in at.metric if m.label.startswith("Trusts more likely than not below"))
    assert int(above.value) >= int(below.value)
    at.sidebar.radio[0].set_value("Emergency admissions via Type 1 A&E").run()
    assert not at.exception
