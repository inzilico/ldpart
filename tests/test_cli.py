"""
The command line interface end to end: every method partitions a small
HDF5 matrix, the two-column output covers all SNPs, the block count is
reported, the default method is the fast determinant, and a missing
input exits with an error.
"""

import numpy as np
import pytest

from ldpart.cli import main
from conftest import block_ld_matrix


@pytest.fixture(scope="module")
def matrix_file(tmp_path_factory):
    h5py = pytest.importorskip("h5py")
    m = block_ld_matrix(60, block_len=10, r=0.9, seed=42)
    path = tmp_path_factory.mktemp("cli") / "ld.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("r2", data=m.astype(np.float32))
    return str(path), m


@pytest.mark.parametrize("args,expect_blocks", [
    (["-m", "fast", "-d", "0.001"], None),
    (["-m", "slogdet", "-d", "0.001"], None),
    (["-m", "condition", "-c", "1000"], None),
    (["-m", "vif", "-v", "10"], None),
    (["-m", "reciprocal", "-r", "5"], None),
    (["-m", "pairwise", "-p", "0.8"], None),
])
def test_methods(matrix_file, tmp_path, args, expect_blocks, capsys):
    path, m = matrix_file
    out = tmp_path / "labels.txt"
    main(["-i", path, "-o", str(out)] + args)
    table = np.loadtxt(out)
    assert table.shape == (m.shape[0], 2)
    assert np.array_equal(table[:, 0], np.arange(m.shape[0]))
    # The reported count matches the labels
    reported = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("Number of blocks")][0]
    nblocks = int(reported.split(":")[1])
    assert len(np.unique(table[:, 1])) == nblocks


def test_default_is_fast(matrix_file, tmp_path, capsys):
    from ldpart import partition_by_determinant
    path, m = matrix_file
    out = tmp_path / "default.txt"
    main(["-i", path, "-o", str(out)])
    expected, nb = partition_by_determinant(m.astype(np.float32),
                                            min_det=0.001)
    assert np.array_equal(np.loadtxt(out), np.asarray(expected))


def test_missing_input(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["-i", str(tmp_path / "nope.h5"),
              "-o", str(tmp_path / "out.txt")])
    assert exc.value.code == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "ldpart" in capsys.readouterr().out
