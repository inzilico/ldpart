"""
Implementation equivalences: the fast Schur determinant agrees with the
LAPACK slogdet reference on NaN free matrices, an open h5py dataset
gives the same labels as the in-memory numpy array (the matrix is
streamed, never copied), and the package version is exposed.
"""

import importlib.metadata

import numpy as np
import pytest

import ldpart
from ldpart import (partition_by_determinant,
                    partition_by_determinant_slogdet)


@pytest.mark.parametrize("theta", [0.1, 0.01, 0.001])
def test_fast_matches_slogdet(matrices, theta):
    for m in matrices["corr"] + matrices["blocks"]:
        fast, nb1 = partition_by_determinant(m, min_det=theta)
        ref, nb2 = partition_by_determinant_slogdet(m, min_det=theta)
        assert nb1 == nb2
        assert np.array_equal(np.asarray(fast), np.asarray(ref))


def test_h5_matches_numpy(matrices, tmp_path):
    h5py = pytest.importorskip("h5py")
    m = matrices["blocks"][1]
    path = tmp_path / "m.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("r2", data=m.astype(np.float32))
    with h5py.File(path, "r") as f:
        from_h5, nb1 = partition_by_determinant(f["r2"],
                                                min_det=0.001)
        from_mem, nb2 = partition_by_determinant(m.astype(np.float32),
                                                 min_det=0.001)
    assert nb1 == nb2
    assert np.array_equal(np.asarray(from_h5), np.asarray(from_mem))


def test_version():
    assert ldpart.__version__ == importlib.metadata.version("ldpart")
