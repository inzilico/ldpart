"""
Assisting functions of the ldpart command line interface: the LD
matrix is read straight from its HDF5 dataset (no copy into memory),
and the small file and folder checks live here. Trimmed from the
helpers of the ld-matrix-partition-01 project, pandas free.
"""

import os
import sys
import time
from typing import Union

import h5py


def attach_matrix(file_obj: h5py.File) -> Union[h5py.Dataset,
                                                h5py.Group]:
    """
    The square matrix of the first object of an open HDF5 file: a
    plain dataset, or a group of chunked pandas arrays
    (block0_values).
    """
    key = list(file_obj.keys())[0]
    obj = file_obj[key]
    if isinstance(obj, h5py.Dataset):
        ds_obj = obj
    elif isinstance(obj, h5py.Group):
        ds_obj = obj["block0_values"]
    else:
        print("Unknown type of dataset")
        sys.exit(1)

    n, m = ds_obj.shape[0], ds_obj.shape[1]
    if n != m:
        print("Dataset is not a square matrix")
        sys.exit(1)
    print(f"Dataset shape: {n} x {m}")
    return ds_obj


def check_output_dir(output: str) -> None:
    # Create the folder holding an output path when missing
    dir_name = os.path.dirname(output)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def check_input_files(files: list) -> None:
    for file in files:
        if file and not os.path.isfile(file):
            print(f"{file} doesn't exist")
            sys.exit(1)


def show_time_elapsed(ts: float) -> None:
    dur = time.strftime("%H:%M:%S", time.gmtime(time.time() - ts))
    print("Time spent:", dur)
