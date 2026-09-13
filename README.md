# ldpart

Certified multicollinearity partitioning of LD matrices: group SNPs
into contiguous blocks of correlated SNPs with greedy criteria built
on classical multicollinearity diagnostics.

Every saved block is **certified** — its submatrix of the LD matrix
provably satisfies the criterion — and the monotone criteria
(determinant, condition number, maximal VIF) additionally certify that
every block is **maximal**: no neighbouring SNP could join it without
violating the criterion. NaN values never extend a block.

## Criteria

| function | method | criterion | certified | maximal | cost per candidate |
|---|---|---|---|---|---|
| `partition_by_determinant` | `fast` | det(block) > θ (Schur updates) | yes | yes (det is monotone under growth) | O(k²) |
| `partition_by_determinant_slogdet` | `slogdet` | det(block) > θ (LAPACK reference) | yes | yes | O(k³) |
| `partition_by_condition` | `condition` | κ(block) = λmax/\|λmin\| ≤ θ | yes | yes (Cauchy interlacing) | O(k³) |
| `partition_by_vif` | `vif` | max VIF = max diag(R⁻¹) ≤ θ | yes | yes (nested regressions) | O(k³) |
| `partition_by_reciprocal_eigenvalues` | `reciprocal` | Σ 1/λq ≤ θ·k | yes | no (the ratio can recover) | O(k³) |
| `partition_by_pairwise` | `pairwise` | max r2(candidate, member) ≥ θ | yes (pairwise sense) | — marginal screen | O(k) |

Classical operating points: det θ = 0.001, κ = 1000 (squared Belsley
CN ≈ 31.6), VIF 5/10, Σ1/λ ≈ 5·k (Chatterjee & Price), pairwise
r² = 0.8.

The determinant, condition, VIF and reciprocal criteria certify that
a block is **not too jointly redundant** (joint criteria); the
pairwise criterion groups SNPs that **are** in LD (marginal screen).
Their cuts land in different places — the joint criteria cut inside
strong LD where redundancy accumulates, the pairwise criterion cuts
where LD has decayed.

## Install

```bash
pip install ldpart
```

## Python API

The input is a square symmetric LD matrix (r or r2, 1 on the
diagonal, SNPs ordered as the rows and columns) — a numpy array or an
open h5py dataset (the matrix is streamed row by row, never copied).

```python
import h5py
from ldpart import partition_by_determinant, labels_to_blocks

with h5py.File("ld.h5", "r") as f:
    matrix = f["r2"]
    labels, nblocks = partition_by_determinant(matrix, min_det=0.001)

blocks = labels_to_blocks(labels)   # [(block, [snp indexes]), ...]
```

Every function returns `(labels, nblocks)`, where labels is a per-SNP
list of `[index, block]` with blocks numbered from 1.

## Command line

```bash
ldpart -i ld.h5 -o labels.txt                       # det 0.001
ldpart -i ld.h5 -o labels.txt -m condition -c 1000
ldpart -i ld.h5 -o labels.txt -m vif -v 10
ldpart -i ld.h5 -o labels.txt -m reciprocal -r 5
ldpart -i ld.h5 -o labels.txt -m pairwise -p 0.8
```

The output has two columns, the zero-based SNP index and the block
label.

## Semantics

- **Greedy growth:** the block starting at SNP i1 is extended by the
  candidate SNP j only while the criterion holds over the window
  matrix[i1:j+1, i1:j+1] *including the candidate*; the first failing
  candidate seeds the next block, so blocks are contiguous and every
  block satisfies its criterion at its final size and at every prefix.
- **Certified maximality:** the determinant can only decrease when a
  SNP is added (Schur complement: each step multiplies it by
  1 − R² ≤ 1), the condition number cannot recover (Cauchy
  interlacing), the maximal VIF cannot recover (nested regressions) —
  so the greedy cut yields the largest admissible block. The sum of
  reciprocal eigenvalues per window size can recover, so its blocks
  are certified but not maximal.
- **NaN semantics:** a NaN anywhere in the candidate window (or the
  candidate's row segment for the pairwise screen) fails the
  extension — NaN never silently extends a block. A SNP with a NaN
  diagonal stays a singleton seed.

## Testing

```bash
pip install -e ".[test]"
pytest tests
```

The suite holds property tests on random correlation matrices:
certification of every block by independent LAPACK recomputation,
maximality of the monotone criteria against brute force, the NaN
semantics, fast/slogdet agreement, numpy/h5py equivalence, and the
command line interface end to end.

## References

- Rogers & Huff (2008), *Heredity* — the r² estimator used in the
  companion study.
- Belsley, Kuh & Welsch (1980) — regression diagnostics, condition
  indices.
- Chatterjee & Price (1977); Dillon & Goldstein (1984) — the
  sum-of-reciprocal-eigenvalues rule.
- The method is developed and validated in the
  ld-matrix-partition-01 study (in preparation).

## License

MIT © Gennady Khvorykh
