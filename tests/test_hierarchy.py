import numpy as np
import pytest

from nhsae.hierarchy import Hierarchy, shrink_cov

METHODS = ["bu", "ols", "wls_struct", "wls_var", "mint_shrink"]


@pytest.fixture(scope="module")
def H(hier):
    return Hierarchy(hier)


@pytest.fixture(scope="module")
def resid(H):
    rng = np.random.default_rng(0)
    b = rng.normal(0, 50, size=(60, H.m))
    return b @ H.S.T + rng.normal(0, 30, size=(60, H.n))


def test_summing_matrix(H):
    assert H.S.shape == (164, 120)
    assert H.S[0].sum() == 120  # England = every trust
    assert (H.S[-120:] == np.eye(120)).all()
    for lvl in ["Region", "ICB"]:
        assert (H.S[H.levels == lvl].sum(axis=0) == 1).all()  # each trust in exactly one node per level


@pytest.mark.parametrize("method", METHODS + ["td"])
def test_reconciled_forecasts_are_coherent(H, resid, method):
    rng = np.random.default_rng(1)
    base = rng.uniform(1000, 5000, size=(12, H.n))  # deliberately incoherent
    hist = np.abs(rng.normal(3000, 300, size=(12, H.m))) @ H.S.T
    G = H.G(method, residuals=resid, history=hist)
    y = H.reconcile(base, G)
    assert np.allclose(y[:, 0], y[:, -H.m:].sum(axis=1))
    for lvl in ["Region", "ICB"]:
        rows = H.S[H.levels == lvl]
        assert np.allclose(y[:, H.levels == lvl], y[:, -H.m:] @ rows.T)


@pytest.mark.parametrize("method", METHODS)
def test_unbiased_methods_leave_coherent_forecasts_unchanged(H, resid, method):
    # S G S = S: a forecast that already adds up is a fixed point of the projection
    b = np.random.default_rng(2).uniform(100, 500, size=(3, H.m))
    coherent = b @ H.S.T
    G = H.G(method, residuals=resid)
    assert np.allclose(H.reconcile(coherent, G), coherent, rtol=1e-6)


def test_shrunk_covariance_is_positive_definite(resid):
    W = shrink_cov(resid)
    assert np.allclose(W, W.T)
    assert np.linalg.eigvalsh(W).min() > 0
