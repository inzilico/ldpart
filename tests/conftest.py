"""
Shared fixtures of the ldpart test suite: random positive definite
correlation matrices and block-structured LD matrices, the substrates
of the property tests.
"""

import numpy as np
import pytest


def random_correlation(n: int, seed: int, samples: int = None) -> np.ndarray:
    """
    A well conditioned positive definite correlation matrix: the
    sample correlation of `samples` standard normal draws (more draws
    than variables, so the matrix is PD with probability 1).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples or n + 20, n))
    # corrcoef collapses to a 0-d array when there is a single
    # variable; atleast_2d restores the (1, 1) correlation matrix
    return np.atleast_2d(np.corrcoef(x, rowvar=False))


def block_ld_matrix(n: int, block_len: int, r: float, seed: int,
                    jitter: float = 0.05) -> np.ndarray:
    """
    An r2 matrix of independent AR(1) blocks: r^(2|i-j|) within every
    block of `block_len` SNPs, a small jitter between blocks. Strong
    within-block LD makes every criterion cut, so the partitions
    exercise real block structure. The jitter keeps the matrix
    positive definite; a final eigenvalue clip repairs the borderline
    cases and the matrix is rescaled to a correlation one.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    within = (i // block_len)[:, None] == (i // block_len)[None, :]
    d = np.abs(i[:, None] - i[None, :])
    m = np.where(within, r ** (2 * d),
                 rng.normal(scale=jitter, size=(n, n)))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    lam, q = np.linalg.eigh(m)
    lam = np.clip(lam, 1e-6, None)
    m = q @ np.diag(lam) @ q.T
    scale = np.sqrt(np.outer(np.diag(m), np.diag(m)))
    m = m / scale
    np.fill_diagonal(m, 1.0)
    return m


@pytest.fixture(scope="session")
def matrices():
    """
    A battery of fixed-seed substrates: well conditioned correlations
    of several sizes and block-structured LD matrices with cut
    patterns for every criterion.
    """
    return {"corr": [random_correlation(n, seed=100 + n)
                     for n in (1, 2, 5, 20, 60)],
            "blocks": [block_ld_matrix(n, block_len, r, seed=200 + n)
                       for n, block_len, r in
                       ((40, 8, 0.95), (60, 10, 0.9), (80, 20, 0.85),
                        (50, 5, 0.8))]}
