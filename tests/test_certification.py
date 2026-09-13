"""
Certification property: every saved block satisfies its criterion by
independent LAPACK recomputation on the block submatrix - the
guarantee the package sells, checked on random and block-structured
correlation matrices.
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

TOL = 1 + 1e-9


def blocks_of(labels, n):
    arr = np.asarray(labels)
    starts = np.flatnonzero(np.diff(arr[:, 1], prepend=arr[0, 1] - 1))
    return [(int(a), int(b)) for a, b in
            zip(starts, np.append(starts[1:], n))]


@pytest.mark.parametrize("theta", [0.1, 0.01, 0.001])
@pytest.mark.parametrize("fun", [partition_by_determinant,
                                 partition_by_determinant_slogdet])
def test_det_certified(matrices, fun, theta):
    for m in matrices["corr"] + matrices["blocks"]:
        labels, _ = fun(m, min_det=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            sub = m[a:b, a:b]
            sign, logdet = np.linalg.slogdet(sub)
            assert sign > 0
            assert np.exp(logdet) > theta / TOL


@pytest.mark.parametrize("theta", [100.0, 1000.0])
def test_condition_certified(matrices, theta):
    for m in matrices["corr"] + matrices["blocks"]:
        labels, _ = partition_by_condition(m, max_cond=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            eig = np.linalg.eigvalsh(m[a:b, a:b])
            assert eig[0] > 0
            assert eig[-1] / abs(eig[0]) <= theta * TOL


@pytest.mark.parametrize("theta", [5.0, 10.0])
def test_vif_certified(matrices, theta):
    for m in matrices["corr"] + matrices["blocks"]:
        labels, _ = partition_by_vif(m, max_vif=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            lam, q = np.linalg.eigh(m[a:b, a:b])
            assert lam[0] > 0
            vif = (q * q / lam).sum(axis=1).max()
            assert vif <= theta * TOL


@pytest.mark.parametrize("theta", [2.0, 5.0])
def test_reciprocal_certified(matrices, theta):
    # Certified at the final size AND at every prefix
    for m in matrices["corr"] + matrices["blocks"]:
        labels, _ = partition_by_reciprocal_eigenvalues(m,
                                                        max_ratio=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            for end in range(a + 1, b + 1):
                lam = np.linalg.eigvalsh(m[a:end, a:end])
                assert lam[0] > 0
                assert (1.0 / lam).sum() <= theta * end * TOL


@pytest.mark.parametrize("theta", [0.5, 0.8])
def test_pairwise_certified(matrices, theta):
    for m in matrices["corr"] + matrices["blocks"]:
        labels, _ = partition_by_pairwise(m, min_r2=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            for j in range(a + 1, b):
                assert np.max(m[j, a:j]) >= theta / TOL
