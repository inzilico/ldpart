"""
ldpart: certified multicollinearity partitioning of LD matrices.

Group SNPs into contiguous blocks of correlated SNPs from a square LD
matrix (r or r2 with 1 on the diagonal, SNPs ordered as the rows and
columns), with six greedy criteria built on classical
multicollinearity diagnostics: the determinant (fast Schur updates
and the LAPACK slogdet reference), the condition number, the maximal
variance inflation factor, the sum of reciprocal eigenvalues, and the
marginal pairwise r2 screen. Every saved block is certified: it
satisfies its criterion, and the monotone criteria additionally
certify maximality. See partitioner for the semantics.
"""

from .partitioner import (
    partition_by_determinant,
    partition_by_determinant_slogdet,
    partition_by_condition,
    partition_by_vif,
    partition_by_reciprocal_eigenvalues,
    partition_by_pairwise,
    labels_to_blocks,
)

__version__ = "0.1.0"

__all__ = [
    "partition_by_determinant",
    "partition_by_determinant_slogdet",
    "partition_by_condition",
    "partition_by_vif",
    "partition_by_reciprocal_eigenvalues",
    "partition_by_pairwise",
    "labels_to_blocks",
    "__version__",
]
