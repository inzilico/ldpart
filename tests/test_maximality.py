"""
Maximality property of the monotone criteria: a greedy block cannot be
extended - the criterion, recomputed by brute force on the window that
includes the next SNP, fails. Checked for the determinant (both
implementations), the condition number and the maximal VIF; the
reciprocal criterion is exempt by design (its ratio to the window
size can recover, so its blocks are certified but not maximal).
"""

import numpy as np
import pytest

from ldpart import (
    partition_by_determinant,
    partition_by_determinant_slogdet,
    partition_by_condition,
    partition_by_vif,
)
from test_certification import blocks_of


def cannot_extend_det(m, a, b, theta):
    if b >= m.shape[0]:
        return True
    sign, logdet = np.linalg.slogdet(m[a:b + 1, a:b + 1])
    return sign <= 0 or np.exp(logdet) <= theta


def cannot_extend_cond(m, a, b, theta):
    if b >= m.shape[0]:
        return True
    eig = np.linalg.eigvalsh(m[a:b + 1, a:b + 1])
    return eig[0] <= 0 or eig[-1] / abs(eig[0]) > theta


def cannot_extend_vif(m, a, b, theta):
    if b >= m.shape[0]:
        return True
    lam, q = np.linalg.eigh(m[a:b + 1, a:b + 1])
    if lam[0] <= 0:
        return True
    return (q * q / lam).sum(axis=1).max() > theta


@pytest.mark.parametrize("theta", [0.1, 0.01, 0.001])
@pytest.mark.parametrize("fun", [partition_by_determinant,
                                 partition_by_determinant_slogdet])
def test_det_maximal(matrices, fun, theta):
    for m in matrices["blocks"]:
        labels, _ = fun(m, min_det=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            assert cannot_extend_det(m, a, b, theta)


@pytest.mark.parametrize("theta", [100.0, 1000.0])
def test_condition_maximal(matrices, theta):
    for m in matrices["blocks"]:
        labels, _ = partition_by_condition(m, max_cond=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            assert cannot_extend_cond(m, a, b, theta)


@pytest.mark.parametrize("theta", [5.0, 10.0])
def test_vif_maximal(matrices, theta):
    for m in matrices["blocks"]:
        labels, _ = partition_by_vif(m, max_vif=theta)
        for a, b in blocks_of(labels, m.shape[0]):
            assert cannot_extend_vif(m, a, b, theta)
