"""
Command line interface of ldpart: partition a square LD matrix into
blocks of correlated SNPs by a multicollinearity criterion. The input
file holds the matrix (1 on the diagonal, SNPs ordered as the rows and
columns) in HDF5; the output holds two columns, the zero-based SNP
index and the block label of every SNP, the format consumed by the
reproducibility layer of the ld-matrix-partition-01 project.

Installed as the console script `ldpart`:

    ldpart -i LD.h5 -o labels.txt -m fast -d 0.001
    ldpart -i LD.h5 -o labels.txt -m condition -c 1000
    ldpart -i LD.h5 -o labels.txt -m vif -v 10
    ldpart -i LD.h5 -o labels.txt -m reciprocal -r 5
    ldpart -i LD.h5 -o labels.txt -m pairwise -p 0.8
"""

import argparse
import time

import numpy as np
import h5py

from . import __version__
from .helpers import (attach_matrix, check_input_files,
                      check_output_dir, show_time_elapsed)
from .partitioner import (
    partition_by_determinant,
    partition_by_determinant_slogdet,
    partition_by_condition,
    partition_by_vif,
    partition_by_reciprocal_eigenvalues,
    partition_by_pairwise,
)

# Method -> (partition function, criterion argument, printed label)
METHODS = {
    "fast": (partition_by_determinant, "min_det", "Minimal determinant"),
    "slogdet": (partition_by_determinant_slogdet, "min_det",
                "Minimal determinant"),
    "condition": (partition_by_condition, "max_cond",
                  "Maximal condition number"),
    "vif": (partition_by_vif, "max_vif", "Maximal VIF"),
    "reciprocal": (partition_by_reciprocal_eigenvalues, "max_ratio",
                   "Maximal sum of reciprocal eigenvalues per block "
                   "size"),
    "pairwise": (partition_by_pairwise, "min_r2",
                 "Minimal pairwise r2"),
}


def parser_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ldpart",
        description="Partition a square LD matrix into blocks of "
                    "correlated SNPs by a multicollinearity criterion")
    parser.add_argument("-i", "--input", required=True,
                        help="/path/to/input.h5 with the matrix in "
                             "HDF5 format")
    parser.add_argument("-o", "--output", required=True,
                        help="/path/to/output.txt for the block labels")
    parser.add_argument("-d", "--min_det", type=float, default=0.001,
                        help="minimal determinant of a block submatrix, "
                             "methods fast and slogdet (default 0.001)")
    parser.add_argument("-c", "--max_cond", type=float, default=1000.0,
                        help="maximal condition number of a block "
                             "submatrix, method condition "
                             "(default 1000)")
    parser.add_argument("-v", "--max_vif", type=float, default=10.0,
                        help="maximal variance inflation factor of a "
                             "block submatrix, method vif (default 10)")
    parser.add_argument("-r", "--max_ratio", type=float, default=5.0,
                        help="maximal sum of reciprocal eigenvalues "
                             "per block size, method reciprocal "
                             "(default 5)")
    parser.add_argument("-p", "--min_r2", type=float, default=0.8,
                        help="minimal pairwise r2 between the "
                             "candidate and a block member, method "
                             "pairwise (default 0.8)")
    parser.add_argument("-m", "--method", choices=tuple(METHODS),
                        default="fast",
                        help="partition function: fast, slogdet, "
                             "condition, vif, reciprocal or pairwise "
                             "(default fast)")
    parser.add_argument("--version", action="version",
                        version=f"ldpart {__version__}")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parser_args(argv)
    ts = time.time()

    check_input_files([args.input])
    check_output_dir(args.output)

    print("Input:", args.input)
    print("Output:", args.output)
    print("Method:", args.method)

    fun, arg_name, label = METHODS[args.method]
    params = {arg_name: getattr(args, arg_name)}
    print(label + ":", params[arg_name])

    with h5py.File(args.input, "r") as f:
        matrix = attach_matrix(f)
        labels, nblocks = fun(matrix, **params)

    np.savetxt(args.output, labels, delimiter=" ", fmt="%d")
    print("Number of blocks:", nblocks)
    show_time_elapsed(ts)


if __name__ == "__main__":
    main()
