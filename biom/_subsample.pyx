# -----------------------------------------------------------------------------
# Copyright (c) 2011-2017, The BIOM Format Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

import numpy as np
cimport numpy as cnp

def _subsample_with_replacement(arr, n, rng):
    """Subsample non-zero values of a sparse array with replacement

    Parameters
    ----------
    arr : {csr_matrix, csc_matrix}
        A 1xM sparse vector
    n : int
        Number of items to subsample from `arr`
    rng : Generator instance
        A random generator. This will likely be an instance returned 
        by np.random.default_rng

    Returns
    -------
    ndarray
        Subsampled data

    Notes
    -----
    This code was adapted from scikit-bio (`skbio.math._subsample`)

    """
    cdef:
        cnp.int64_t counts_sum
        cnp.ndarray[cnp.float64_t, ndim=1] data = arr.data
        cnp.ndarray[cnp.int32_t, ndim=1] indptr = arr.indptr
        Py_ssize_t i, length

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start
        counts_sum = data[start:end].sum()
        
        pvals = data[start:end] / counts_sum
        data[start:end] = rng.multinomial(n, pvals)


def _subsample_without_replacement(arr, n, rng):
    """Subsample non-zero values of a sparse array w/out replacement

    Parameters
    ----------
    arr : {csr_matrix, csc_matrix}
        A 1xM sparse vector
    n : int
        Number of items to subsample from `arr`
    rng : Generator instance
        A random generator. This will likely be an instance returned 
        by np.random.default_rng

    Returns
    -------
    ndarray
        Subsampled data

    Notes
    -----
    This code was adapted from scikit-bio (`skbio.math._subsample`)

    """
    cdef:
        cnp.int64_t counts_sum
        cnp.ndarray[cnp.float64_t, ndim=1] data = arr.data
        cnp.ndarray[cnp.int64_t, ndim=1] data_i = arr.data.astype(np.int64)
        cnp.ndarray[cnp.float64_t, ndim=1] result
        cnp.ndarray[cnp.int32_t, ndim=1] indptr = arr.indptr
        cnp.ndarray[cnp.int32_t, ndim=1] permuted, unpacked, r
        Py_ssize_t i, length

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start
        counts_sum = data[start:end].sum()
        
        if counts_sum < n:
            data[start:end] = 0
            continue

        r = np.arange(length, dtype=np.int32)
        unpacked = np.repeat(r, data_i[start:end])
        permuted = rng.permutation(unpacked)[:n]

        result = np.zeros(length, dtype=np.float64)
        for idx in range(permuted.shape[0]):
            result[permuted[idx]] += 1

        data[start:end] = result


def _subsample(arr, n, with_replacement, rng):
    """Subsample non-zero values of a sparse array

    Parameters
    ----------
    arr : {csr_matrix, csc_matrix}
        A 1xM sparse vector
    n : int
        Number of items to subsample from `arr`
    with_replacement : bool
        Whether to permute or use multinomial sampling
    rng : Generator instance
        A random generator. This will likely be an instance returned 
        by np.random.default_rng

    Returns
    -------
    ndarray
        Subsampled data

    Notes
    -----
    This code was adapted from scikit-bio (`skbio.math._subsample`)

    """
    if (with_replacement):
       return _subsample_with_replacement(arr, n, rng)
    else:
       return _subsample_without_replacement(arr, n, rng)
