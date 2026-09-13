"""
Label format properties shared by every criterion: the labels cover
all SNPs in order, the blocks are numbered from 1 without gaps, and
every block is a contiguous range of SNP indexes.
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

CASES = [
    (partition_by_determinant, dict(min_det=0.001)),
    (partition_by_determinant_slogdet, dict(min_det=0.001)),
    (partition_by_condition, dict(max_cond=1000)),
    (partition_by_vif, dict(max_vif=10)),
    (partition_by_reciprocal_eigenvalues, dict(max_ratio=5)),
    (partition_by_pairwise, dict(min_r2=0.8)),
]


@pytest.mark.parametrize("fun,params", CASES)
def test_label_format(matrices, fun, params):
    for m in matrices["corr"] + matrices["blocks"]:
        labels, nblocks = fun(m, **params)
        arr = np.asarray(labels)
        n = m.shape[0]
        assert arr.ndim == 2 and arr.shape == (n, 2)
        assert np.array_equal(arr[:, 0], np.arange(n))
        blocks = arr[:, 1]
        assert blocks.min() == 1 and blocks.max() == nblocks
        assert len(np.unique(blocks)) == nblocks
        # Contiguity: every block is one range of indexes, and the
        # block ids grow with the SNP order
        assert np.all(np.diff(blocks) >= 0)
        for b in range(1, nblocks + 1):
            idx = np.flatnonzero(blocks == b)
            assert idx[-1] - idx[0] == len(idx) - 1


@pytest.mark.parametrize("fun,params", CASES)
def test_determinism(matrices, fun, params):
    m = matrices["blocks"][1]
    first, nb1 = fun(m, **params)
    second, nb2 = fun(m, **params)
    assert nb1 == nb2
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_empty_and_single(matrices):
    for m in matrices["corr"][:2]:
        labels, nblocks = partition_by_determinant(m, min_det=0.001)
        assert nblocks == 1 if m.shape[0] > 1 else nblocks in (0, 1)
        assert len(labels) == m.shape[0]
