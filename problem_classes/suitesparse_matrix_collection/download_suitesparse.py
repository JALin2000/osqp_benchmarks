"""
Python equivalent of download.jl

Downloads SuiteSparse 'least squares problem' matrices and saves them in the
HDF5 format expected by suitesparse_lasso.py:
    /b        – dense float64 vector
    /A/ir     – CSC row indices
    /A/jc     – CSC column pointers
    /A/data   – CSC float64 values

Requirements (already in rlqp env):
    ssgetpy   – pip install ssgetpy   (already done)
    scipy     – for loadmat
    tables    – for writing HDF5
    numpy

Usage:
    conda run -n rlqp python download_suitesparse.py
"""

import os
import shutil
import tempfile

import numpy as np
import scipy.io
import scipy.sparse as spa
import tables
import ssgetpy

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mat_v5(filepath):
    """Load Problem.A and Problem.b from a SuiteSparse v5 .mat file."""
    data = scipy.io.loadmat(filepath)
    prob = data['Problem'][0, 0]

    A = spa.csc_matrix(prob['A'])

    if 'b' in prob.dtype.names and prob['b'].size > 0:
        b = np.array(prob['b']).flatten().astype(np.float64)
        b_source = 'stored'
    else:
        # Replicate Julia: b = A * x0 + s0  (random, fixed seed per matrix)
        m, n = A.shape
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal(n)
        s0 = rng.standard_normal(m)
        b = A @ x0 + s0
        b_source = 'random'

    return A, b, b_source


def save_hdf5(filepath, A_csc, b):
    """Write A (CSC) and b in HDF5 format compatible with suitesparse_lasso.py."""
    with tables.open_file(filepath, 'w') as f:
        f.create_array('/', 'b', b.astype(np.float64))
        g = f.create_group('/', 'A')
        f.create_array(g, 'ir', A_csc.indices.astype(np.int32))
        f.create_array(g, 'jc', A_csc.indptr.astype(np.int32))
        f.create_array(g, 'data', A_csc.data.astype(np.float64))


def main():
    results = ssgetpy.search(kind='least squares problem')
    print(f"Found {len(results)} 'least squares problem' matrices\n")

    prob_count = 0
    skip_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for matrix in results:
            safe_name = f"{matrix.group}_{matrix.name}"
            out_path = os.path.join(OUTPUT_DIR, safe_name + '.mat')

            if os.path.exists(out_path):
                print(f"  [skip]  {safe_name}  (already exists)")
                skip_count += 1
                prob_count += 1
                continue

            print(f"  Downloading  {matrix.group}/{matrix.name} "
                  f"({matrix.rows}×{matrix.cols}, nnz={matrix.nnz}) ...",
                  end='  ', flush=True)

            try:
                result = matrix.download(format='MAT', destpath=tmpdir, extract=True)
                # download() returns (tarball_path, mat_path) or just mat_path
                if isinstance(result, tuple):
                    mat_path = result[1]
                else:
                    mat_path = result
            except Exception as e:
                print(f"DOWNLOAD FAILED: {e}")
                continue

            try:
                A, b, b_source = load_mat_v5(mat_path)
            except Exception as e:
                print(f"LOAD FAILED: {e}")
                continue

            A = spa.csc_matrix(A)
            m, n = A.shape

            try:
                save_hdf5(out_path, A, b)
            except Exception as e:
                print(f"SAVE FAILED: {e}")
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            print(f"saved  ({m}×{n}, nnz={A.nnz}, b={b_source})")
            prob_count += 1

            # Clean up downloaded raw file
            try:
                os.remove(mat_path)
            except OSError:
                pass

    print(f"\nDone. Total: {prob_count} matrices ({skip_count} skipped, already on disk).")


if __name__ == '__main__':
    main()
