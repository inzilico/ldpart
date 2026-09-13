"""
NaN semantics: a NaN never extends a block. For the joint criteria no
block submatrix may contain a NaN; for the pairwise screen the
candidate's row segment must be NaN free; a SNP with a NaN diagonal
stays a singleton. Also the degenerate shapes: the identity matrix
(one block for the joint criteria, singletons for the pairwise
screen) and a perfectly correlated matrix (singletons for every joint
criterion, one block for the pairwise screen).
"""

import numpy as np
import pytest

from ldpart import (
    partition_by_determinant,
    partition_by_determinant_slogdet,
    partition_by_condition,
    partition_by_vif,
    partition_by_reciprocal_eigenvalues,
    partition_by_pairwise,
)
from test_certification import blocks_of

JOINT = [
    (partition_by_determinant, dict(min_det=0.001)),
    (partition_by_determinant_slogdet, dict(min_det=0.001)),
    (partition_by_condition, dict(max_cond=1000)),
    (partition_by_vif, dict(max_vif=10)),
    (partition_by_reciprocal_eigenvalues, dict(max_ratio=5)),
]


def with_nans(m, seed, share=0.08):
    rng = np.random.default_rng(seed)
    out = m.astype(float).copy()
    mask = rng.random(out.shape) < share
    mask = np.triu(mask, 1)
    out[mask] = np.nan
    out[mask.T] = np.nan
    return out


@pytest.mark.parametrize("fun,params", JOINT)
def test_nan_never_extends(matrices, fun, params):
    for k, m in enumerate(matrices["blocks"]):
        noisy = with_nans(m, seed=300 + k)
        labels, _ = fun(noisy, **params)
        for a, b in blocks_of(labels, noisy.shape[0]):
            assert not np.isnan(noisy[a:b, a:b]).any()


def test_nan_pairwise(matrices):
    m = with_nans(matrices["blocks"][0], seed=301)
    labels, _ = partition_by_pairwise(m, min_r2=0.8)
    for a, b in blocks_of(labels, m.shape[0]):
        for j in range(a + 1, b):
            assert not np.isnan(m[j, a:j]).any()


def test_nan_diagonal_singleton():
    n = 30
    m = np.eye(n)
    m[7, 7] = np.nan
    for fun, params in JOINT:
        labels, nblocks = fun(m, **params)
        arr = np.asarray(labels)
        assert arr[7, 1] != arr[6, 1] and arr[7, 1] != arr[8, 1]


def test_all_nan_singletons():
    n = 12
    m = np.full((n, n), np.nan)
    np.fill_diagonal(m, 1.0)
    for fun, params in JOINT:
        labels, nblocks = fun(m, **params)
        assert nblocks == n
    labels, nblocks = partition_by_pairwise(m, min_r2=0.8)
    assert nblocks == n


def test_identity_matrix():
    n = 25
    m = np.eye(n)
    _, nb_det = partition_by_determinant(m, min_det=0.001)
    _, nb_cond = partition_by_condition(m, max_cond=1000)
    _, nb_vif = partition_by_vif(m, max_vif=10)
    _, nb_recip = partition_by_reciprocal_eigenvalues(m, max_ratio=2)
    _, nb_pair = partition_by_pairwise(m, min_r2=0.8)
    assert (nb_det, nb_cond, nb_vif, nb_recip) == (1, 1, 1, 1)
    assert nb_pair == n


def test_perfect_ld_matrix():
    n = 15
    m = np.ones((n, n))
    _, nb_det = partition_by_determinant(m, min_det=0.001)
    _, nb_cond = partition_by_condition(m, max_cond=1000)
    _, nb_vif = partition_by_vif(m, max_vif=10)
    _, nb_recip = partition_by_reciprocal_eigenvalues(m, max_ratio=5)
    _, nb_pair = partition_by_pairwise(m, min_r2=0.8)
    assert (nb_det, nb_cond, nb_vif, nb_recip) == (n, n, n, n)
    assert nb_pair == 1
