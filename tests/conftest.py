import pandas as pd
import pytest

from nhsae.config import HIERARCHY, PANEL


@pytest.fixture(scope="session")
def panel():
    return pd.read_csv(PANEL, parse_dates=["month"])


@pytest.fixture(scope="session")
def hier():
    return pd.read_csv(HIERARCHY)
