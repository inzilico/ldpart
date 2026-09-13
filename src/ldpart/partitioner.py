"""
Library of functions to group SNPs into blocks of correlated SNPs.

The input is a square symmetric LD matrix (r or r2 with 1 on the diagonal,
SNPs ordered as the rows/columns), either a numpy array or an h5py dataset.
Every grouping function returns (labels, nblocks), where labels is a per-SNP
list [index, block_label] with blocks numbered from 1. Each SNP gets exactly
one label, and blocks are contiguous ranges of indices. The two-column labels
can be saved and passed to 2hlist-02.py (det-01 project) to build *.hlist
files for haplotype inference and testing in PLINK.

Functions
---------
partition_by_determinant         determinant criterion, incremental Schur
                                 complement updates (fast)
partition_by_determinant_slogdet determinant criterion, full LAPACK slogdet
                                 of the growing window at each step
                                 (reference implementation)
partition_by_condition           condition number criterion (squared
                                 Belsley condition number), full LAPACK
                                 eigvalsh of the growing window
partition_by_vif                 maximal variance inflation factor (VIF)
                                 criterion, per-window eigendecomposition
partition_by_reciprocal_eigenvalues  sum of reciprocal eigenvalues criterion
                                 (= sum of VIF = trace of the inverse)
partition_by_pairwise            pairwise r2 with the block (classic
                                 |r| > 0.8 screen), incremental row maximum
labels_to_blocks                 convert labels into a list of blocks

Corrected semantics (see det-01/audit-partition-ld-03.md)
---------------------------------------------------------
Unlike det-01/partition-ld-03.py, the candidate SNP is always part of the
tested submatrix: the block starting at i1 is extended by SNP j only if
det(matrix[i1:j+1, i1:j+1]) > min_det with a positive sign. The first
candidate violating the condition seeds the next block, so every saved
block is certified: the determinant of its own submatrix exceeds min_det.
NaN values are treated as a failure to extend, never as an automatic pass.

All functions share the same greedy mechanism (grow the block, the first
failing candidate seeds the next one, NaN never extends a block) and the
same certified-block semantics; they differ in the tested criterion. Two
cut directions coexist: the determinant, condition number, VIF and
reciprocal eigenvalue criteria certify that a block is not TOO jointly
redundant (a block ends before becoming ill-conditioned), while the
pairwise criterion groups SNPs that ARE in LD (an uncorrelated SNP starts
a new block).

Command line wrapper: run-partition-01.py in this folder.

Part of the ld-matrix-partition-01 project.
Created: September 8, 2026
"""

import numpy as np


def _extend_inverse(inv: np.ndarray, u: np.ndarray, s: float) -> np.ndarray:
    """
    Inverse of the extended matrix [[M, c], [c.T, d]] given the inverse of M,
    u = M^-1 c, and the Schur complement s = d - c.T u (Sherman-Morrison).
    """
    k = u.shape[0]
    w = u / s
    new_inv = np.empty((k + 1, k + 1), dtype=np.float64)
    new_inv[:k, :k] = inv + np.outer(w, u)
    new_inv[:k, k] = -w
    new_inv[k, :k] = -w
    new_inv[k, k] = 1.0 / s
    return new_inv


def _check_matrix(matrix) -> int:
    # Get the size of a square 2D matrix
    shape = matrix.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expected a square matrix")
    return shape[0]


def _seed(matrix, i: int):
    # State of a block that consists of the single SNP i:
    # sign and log|det| of its 1x1 submatrix, and its inverse
    d = float(matrix[i, i])
    if np.isnan(d) or d <= 0:
        # Uncertifiable seed: the block stays a singleton
        return 0, 0.0, np.ones((1, 1))
    return 1, np.log(d), np.array([[1.0 / d]])


def partition_by_determinant(matrix, min_det: float = 0.001):
    """
    Group SNPs into blocks by the determinant of the LD submatrix (fast).

    The block starting at SNP i1 is extended by the candidate SNP j while
    det(matrix[i1:j+1, i1:j+1]) > min_det holds with a positive sign. The
    first candidate violating the condition seeds the next block. Since the
    determinant of a correlation matrix can only decrease when a SNP is
    added (each step multiplies it by 1 - R^2 <= 1), the greedy growth finds
    the maximal block for the criterion.

    The determinant is carried along incrementally with the Schur complement
        det(M') = det(M) * (d - c.T M^-1 c),  M' = [[M, c], [c.T, d]],
    which needs one row of the matrix and O(k^2) work per step, instead of
    a full O(k^3) factorization of the growing window (O(k^4) per block in
    total). The explicit inverse is updated with the Sherman-Morrison
    formula. On ill-conditioned blocks the accumulated value can differ
    from the LAPACK determinant in the last bits; use
    partition_by_determinant_slogdet() for the reference behaviour.

    A NaN anywhere in the candidate's row (or a NaN, zero, or negative
    Schur complement) fails the extension: the candidate starts a new
    block. NaN never silently extends a block.

    Parameters
    ----------
    matrix : h5py.Dataset or numpy.ndarray
        Square symmetric LD matrix (r or r2) with 1 on the diagonal.
    min_det : float
        Minimal determinant of a block submatrix, within (0, 1).

    Returns
    -------
    list
        Labels [index, block] of the matrix elements
    int
        Number of blocks
    """
    n = _check_matrix(matrix)
    if not 0.0 < min_det < 1.0:
        raise ValueError(f"min_det must be within (0, 1), got {min_det}")
    if n == 0:
        return [], 0

    log_min = np.log(min_det)

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0
    sign, logdet, inv = _seed(matrix, 0)

    # The symmetry of the matrix is used to read the candidate's row
    # (contiguous in memory) instead of its column
    for j in range(1, n):
        extend = False
        if sign > 0:
            row = np.asarray(matrix[j, i1:j + 1], dtype=np.float64)
            if not np.isnan(row).any():
                c, d = row[:-1], row[-1]
                u = inv.dot(c)
                s = d - c.dot(u)
                if not np.isnan(s) and s > 0:
                    new_logdet = logdet + np.log(s)
                    if new_logdet > log_min:
                        extend = True
        if extend:
            # The candidate joins the current block
            labels.append([j, b])
            logdet = new_logdet
            inv = _extend_inverse(inv, u, s)
        else:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j
            sign, logdet, inv = _seed(matrix, j)

    return labels, b


def partition_by_determinant_slogdet(matrix, min_det: float = 0.001):
    """
    Group SNPs into blocks by the determinant of the LD submatrix
    (reference implementation).

    Same criterion and semantics as partition_by_determinant(), but the
    determinant of the growing window matrix[i1:j+1, i1:j+1] is recomputed
    with np.linalg.slogdet at every step, which costs O(k^4) flops per
    block. Kept to define the semantics and to verify the fast version.
    Note that LAPACK can return a finite determinant for a window that
    contains a NaN (the entry may stay off the pivoting path), so on a
    matrix with NaNs the fast version, which checks the row explicitly,
    is the authoritative one.
    """
    n = _check_matrix(matrix)
    if not 0.0 < min_det < 1.0:
        raise ValueError(f"min_det must be within (0, 1), got {min_det}")
    if n == 0:
        return [], 0

    log_min = np.log(min_det)

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0

    for j in range(1, n):
        submatrix = np.asarray(matrix[i1:j + 1, i1:j + 1], dtype=np.float64)
        sign, logdet = np.linalg.slogdet(submatrix)
        if (not np.isfinite(logdet)) or sign <= 0 or logdet <= log_min:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j
        else:
            labels.append([j, b])

    return labels, b


def partition_by_condition(matrix, max_cond: float = 1000.0):
    """
    Group SNPs into blocks by the condition number of the LD submatrix.

    The block starting at SNP i1 is extended by the candidate SNP j while
    cond(matrix[i1:j+1, i1:j+1]) <= max_cond, where cond is the spectral
    condition number lambda_max/|lambda_min| from np.linalg.eigvalsh. The
    first candidate violating the condition seeds the next block, so every
    saved block is certified: the condition number of its submatrix, and of
    every prefix of it, does not exceed max_cond.

    By the Cauchy interlacing theorem, appending a row and a column to a
    symmetric matrix can only increase lambda_max and decrease lambda_min,
    so the condition number of the growing window cannot recover after a
    failure: the greedy growth finds the maximal block for the criterion.
    A window that stops being positive definite (lambda_min <= 0) has an
    infinite condition number and always fails, which turns numerical
    singularity into an explicit cut.

    Unlike the determinant, the condition number never underflows: it stays
    meaningful far beyond the sizes where a determinant leaves the float64
    range. Note the convention: the condition indices of Belsley et al. are
    CI_q = sqrt(lambda_max/lambda_q) with CN = max_q CI_q = sqrt(kappa),
    so this function's criterion kappa <= max_cond with the default 1000
    corresponds to the classical CN = sqrt(1000) ~ 31.6, just above the
    severe-multicollinearity level of 30.

    A NaN anywhere in the candidate window fails the extension, never an
    automatic pass. The eigenvalues are recomputed from the window with
    LAPACK at every step (O(k^4) flops per block), mirroring
    partition_by_determinant_slogdet().

    Parameters
    ----------
    matrix : h5py.Dataset or numpy.ndarray
        Square symmetric LD matrix (r or r2) with 1 on the diagonal.
    max_cond : float
        Maximal condition number of a block submatrix, >= 1.

    Returns
    -------
    list
        Labels [index, block] of the matrix elements
    int
        Number of blocks
    """
    n = _check_matrix(matrix)
    if max_cond < 1.0:
        raise ValueError(f"max_cond must be >= 1, got {max_cond}")
    if n == 0:
        return [], 0

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0

    for j in range(1, n):
        submatrix = np.asarray(matrix[i1:j + 1, i1:j + 1], dtype=np.float64)
        if np.isnan(submatrix).any():
            extend = False
        else:
            eig = np.linalg.eigvalsh(submatrix)
            lmin = abs(eig[0])
            cond = eig[-1] / lmin if lmin > 0 else np.inf
            extend = cond <= max_cond
        if extend:
            # The candidate joins the current block
            labels.append([j, b])
        else:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j

    return labels, b


def partition_by_vif(matrix, max_vif: float = 10.0):
    """
    Group SNPs into blocks by the maximal variance inflation factor (VIF)
    of the block submatrix.

    VIF_i = (R^-1)_ii = 1 / (1 - R2_i), where R2_i is the multiple
    correlation of SNP i with the other SNPs of the block. The block
    starting at SNP i1 is extended by the candidate SNP j while
    max_i VIF_i <= max_vif over matrix[i1:j+1, i1:j+1]; the first
    candidate violating the condition seeds the next block, so every saved
    block is certified: the maximal VIF of its own submatrix (and of every
    prefix of it) does not exceed max_vif. The equivalent tolerance
    criterion 1/VIF >= 1/max_vif is the same test with an inverted
    threshold. Classical reference levels: VIF 5 moderate, 10 severe
    multicollinearity (tolerance 0.2 / 0.1).

    Monotonicity: for a positive-definite extension M' = [[M, c], [c.T, d]]
    with u = M^-1 c and Schur complement s = d - c.T u > 0, the updated
    inverse is M'^-1 = [[M^-1 + uu.T/s, -u/s], [-u.T/s, 1/s]], so every
    diagonal entry can only grow ((M'^-1)_ii = (M^-1)_ii + u_i^2/s for the
    old indices, 1/s for the new one): the maximal VIF cannot recover
    after a failure, and the greedy growth finds the maximal block.

    The VIFs are read from the eigendecomposition R = Q diag(lambda) Q.T:
    (R^-1)_ii = sum_q Q[i,q]^2 / lambda_q, computed fresh for every window
    (O(k^4) flops per block), which also provides the positive-definite
    check: a window with lambda_min <= 0 always fails. A NaN anywhere in
    the window fails the extension, never an automatic pass.

    Parameters
    ----------
    matrix : h5py.Dataset or numpy.ndarray
        Square symmetric LD matrix (r or r2) with 1 on the diagonal.
    max_vif : float
        Maximal variance inflation factor of a block submatrix, >= 1.

    Returns
    -------
    list
        Labels [index, block] of the matrix elements
    int
        Number of blocks
    """
    n = _check_matrix(matrix)
    if max_vif < 1.0:
        raise ValueError(f"max_vif must be >= 1, got {max_vif}")
    if n == 0:
        return [], 0

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0

    for j in range(1, n):
        submatrix = np.asarray(matrix[i1:j + 1, i1:j + 1], dtype=np.float64)
        if np.isnan(submatrix).any():
            extend = False
        else:
            lam, q = np.linalg.eigh(submatrix)
            if lam[0] > 0:
                vif = (q * q / lam).sum(axis=1).max()
                extend = vif <= max_vif
            else:
                extend = False
        if extend:
            # The candidate joins the current block
            labels.append([j, b])
        else:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j

    return labels, b


def partition_by_reciprocal_eigenvalues(matrix, max_ratio: float = 5.0):
    """
    Group SNPs into blocks by the sum of reciprocal eigenvalues of the
    block submatrix (Kendall 1957; Silvey 1969).

    The block starting at SNP i1 is extended by the candidate SNP j while
    sum_q 1/lambda_q <= max_ratio * k over the k x k window
    matrix[i1:j+1, i1:j+1]: the classical rule of Chatterjee & Price
    (1977) and Dillon & Goldstein (1984) flags collinearity when the sum
    of reciprocal eigenvalues is about five times the number of variables.
    Since trace(R^-1) = sum_q 1/lambda_q = sum_i VIF_i, this is the
    aggregate version of the VIF criterion, and every saved block is
    certified: the sum does not exceed max_ratio * size at its final size
    nor at any prefix.

    Unlike the determinant, condition number and VIF criteria, the ratio
    to the window size can recover after a failure (an almost uncorrelated
    candidate raises k by 1 but the sum by barely more), so the greedy cut
    is deterministic but the blocks are not certified maximal. A window
    that stops being positive definite (lambda_min <= 0) always fails, a
    NaN anywhere in the window fails the extension.

    The eigenvalues are recomputed from the window with LAPACK at every
    step (O(k^4) flops per block).

    Parameters
    ----------
    matrix : h5py.Dataset or numpy.ndarray
        Square symmetric LD matrix (r or r2) with 1 on the diagonal.
    max_ratio : float
        Maximal sum of reciprocal eigenvalues per window size, >= 1.

    Returns
    -------
    list
        Labels [index, block] of the matrix elements
    int
        Number of blocks
    """
    n = _check_matrix(matrix)
    if max_ratio < 1.0:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
    if n == 0:
        return [], 0

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0

    for j in range(1, n):
        submatrix = np.asarray(matrix[i1:j + 1, i1:j + 1], dtype=np.float64)
        if np.isnan(submatrix).any():
            extend = False
        else:
            lam = np.linalg.eigvalsh(submatrix)
            if lam[0] > 0:
                extend = (1.0 / lam).sum() <= max_ratio * lam.size
            else:
                extend = False
        if extend:
            # The candidate joins the current block
            labels.append([j, b])
        else:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j

    return labels, b


def partition_by_pairwise(matrix, min_r2: float = 0.8):
    """
    Group SNPs into blocks by pairwise LD with the block: the classic
    pairwise correlation screen (absolute r above ~0.8, e.g. Gujarati &
    Porter), applied to an r2 matrix.

    The block starting at SNP i1 is extended by the candidate SNP j while
    max over the block members of r2(candidate, member) >= min_r2. This
    inverts the cut direction of the determinant, condition number, VIF
    and reciprocal eigenvalue criteria: a SNP in LD with the block JOINS
    it, and an uncorrelated SNP starts a new block. Every saved block is
    certified in the pairwise sense: each member after the first has
    r2 >= min_r2 with an earlier member of the same block. Only pairwise
    (marginal) LD is tested, not the joint redundancy of the block.

    The candidate needs one row segment and O(k) work per step, so this is
    the cheapest criterion (comparable to PLINK-style LD pruning). A NaN
    in the candidate's row segment fails the extension.

    Parameters
    ----------
    matrix : h5py.Dataset or numpy.ndarray
        Square symmetric LD matrix (r2, with 1 on the diagonal); for an r
        matrix pass min_r2 as a threshold on r2 = value^2.
    min_r2 : float
        Minimal pairwise r2 between the candidate and a block member,
        within (0, 1].

    Returns
    -------
    list
        Labels [index, block] of the matrix elements
    int
        Number of blocks
    """
    n = _check_matrix(matrix)
    if not 0.0 < min_r2 <= 1.0:
        raise ValueError(f"min_r2 must be within (0, 1], got {min_r2}")
    if n == 0:
        return [], 0

    # Initiate the first block
    b = 1
    labels = [[0, b]]
    i1 = 0

    for j in range(1, n):
        # The symmetry of the matrix is used to read the candidate's row
        # (contiguous in memory) instead of its column
        segment = np.asarray(matrix[j, i1:j], dtype=np.float64)
        if segment.size and not np.isnan(segment).any() and segment.max() >= min_r2:
            # The candidate joins the current block
            labels.append([j, b])
        else:
            # The candidate fails: it seeds the next block
            b += 1
            labels.append([j, b])
            i1 = j

    return labels, b


def labels_to_blocks(labels: list) -> list:
    """
    Convert labels [index, block] into a list of (block, indices) tuples
    ordered by the block number.
    """
    blocks = {}
    for i, b in labels:
        blocks.setdefault(b, []).append(i)
    return [(b, idx) for b, idx in sorted(blocks.items())]
