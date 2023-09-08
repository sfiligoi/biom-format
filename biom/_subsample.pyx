# -----------------------------------------------------------------------------
# Copyright (c) 2011-2017, The BIOM Format Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

import numpy as np
cimport numpy as cnp

cdef _subsample_with_replacement(cnp.ndarray[cnp.float64_t, ndim=1] data,
                                 cnp.ndarray[cnp.int32_t, ndim=1] indptr,
                                 cnp.int64_t n,
                                 object rng):
    """Subsample non-zero values of a sparse array with replacement

    Parameters
    ----------
    data : {csr_matrix, csc_matrix}.data
        A 1xM sparse vector data
    indptr : {csr_matrix, csc_matrix}.indptr
        A 1xM sparse vector indptr
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
        cnp.int32_t start,end,length
        Py_ssize_t i
        cnp.ndarray[cnp.float64_t, ndim=1] pvals

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start
        counts_sum = data[start:end].sum()
        
        pvals = data[start:end] / counts_sum
        data[start:end] = rng.multinomial(n, pvals)


cdef _subsample_without_replacement(cnp.ndarray[cnp.float64_t, ndim=1] data,
                                    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
                                    cnp.int64_t n,
                                    object rng):
    """Subsample non-zero values of a sparse array w/out replacement

    Parameters
    ----------
    data : {csr_matrix, csc_matrix}.data
        A 1xM sparse vector data
    indptr : {csr_matrix, csc_matrix}.indptr
        A 1xM sparse vector indptr
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
        cnp.int64_t counts_sum, count_el, perm_count_el
        cnp.int64_t count_rem
        cnp.ndarray[cnp.int64_t, ndim=1] permuted
        Py_ssize_t i, idx
        cnp.int32_t length,el,start,end

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start
        counts_sum = data[start:end].sum()
        
        if counts_sum < n:
            data[start:end] = 0
            continue

        permuted = rng.choice(counts_sum, n, replace=False, shuffle=False)
        permuted.sort()

        # now need to do reverse mapping
        # since I am not using np.repeat anymore
        # reminder, old logic was
        #   r = np.arange(length)
        #   unpacked = np.repeat(r, data_i[start:end])
        #   permuted_unpacked = rng.choice(unpacked, n, replace=False, shuffle=False)
        # 
        # specifically, what we're going to do here is randomly pick what elements within
        # each sample to keep. this is analogous issuing the prior np.repeat call, and obtaining
        # a random set of index positions for that resulting array. however, we do not need to
        # perform the np.repeat call as we know the length of that resulting vector already,
        # and additionally, we can compute the sample associated with an index in that array
        # without constructing it.

        el = 0         # index in result/data
        count_el = 0  # index in permutted
        count_rem = long(data[start])  # since each data has multiple els, sub count there
        data[start] = 0.0
        for idx in range(n):
            perm_count_el = permuted[idx]
            # the array is sorted, so just jump ahead
            while (perm_count_el - count_el) >= count_rem:
               count_el += count_rem
               el += 1
               count_rem = long(data[start+el])
               data[start+el] = 0.0
            count_rem -= (perm_count_el - count_el)
            count_el = perm_count_el

            data[start+el] += 1
        # clean up tail elements
        data[start+el+1:end] = 0.0


cdef _subsample_fast_t1(cnp.ndarray[cnp.float64_t, ndim=1] data,
                        cnp.ndarray[cnp.int32_t, ndim=1] indptr,
                        cnp.int64_t n,
                        object rng):
    """Subsample non-zero values of a sparse array using an approximate method.

    Parameters
    ----------
    data : {csr_matrix, csc_matrix}.data
        A 1xM sparse vector data
    indptr : {csr_matrix, csc_matrix}.indptr
        A 1xM sparse vector indptr
    n : int
        Number of items to subsample from `arr`
    rng : Generator instance
        A random generator. This will likely be an instance returned
        by np.random.default_rng

    Returns
    -------
    ndarray
        Subsampled data

    """
    cdef:
        cnp.int64_t counts_sum, int_counts_sum
        cnp.int32_t rels,int_counts_diff
        cnp.int32_t start,end,length,j
        Py_ssize_t i
        cnp.int64_t el, eldiv, elrem
        cnp.ndarray[cnp.int64_t, ndim=1] idata
        cnp.ndarray[cnp.int32_t, ndim=1] rems

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start

        idata = data[start:end].astype(np.int64)
        counts_sum = idata.sum()

        # TODO: This could potentially be relaxed
        if counts_sum < n:
            data[start:end] = 0
            continue

        rems = np.empty(length, np.int32)
        int_counts_sum = 0
        rels = 0
        for j in range(length):
            el = idata[j]
            # compute the remainders of proportional vals
            el = n*el
            elrem = el % counts_sum
            if elrem>0:
              # preserve which index had a mon-zero reminder
              rems[rels] = j
              rels += 1
            # keep only the integer part of the proportional val
            eldiv = el // counts_sum
            idata[j] = eldiv
            int_counts_sum += eldiv

        # see how much off we are
        # since it is based on indexes, it will be 32-bit
        int_counts_diff = n-int_counts_sum
        if int_counts_diff>0:
            # randomly pick from the reminders
            # do not care how much the remainder was
            # we always increase by at most one
            rems.resize(rels,refcheck=False)
            rng.shuffle(rems)
            for j in range(int_counts_diff):
                idata[rems[j]] += 1

        data[start:end] = idata


cdef _subsample_fast(cnp.ndarray[cnp.float64_t, ndim=1] data,
                     cnp.ndarray[cnp.int32_t, ndim=1] indptr,
                     cnp.int64_t n,
                     object rng):
    """Subsample non-zero values of a sparse array using an approximate method.

    Parameters
    ----------
    data : {csr_matrix, csc_matrix}.data
        A 1xM sparse vector data
    indptr : {csr_matrix, csc_matrix}.indptr
        A 1xM sparse vector indptr
    n : int
        Number of items to subsample from `arr`
    rng : Generator instance
        A random generator. This will likely be an instance returned
        by np.random.default_rng

    Returns
    -------
    ndarray
        Subsampled data

    """
    cdef:
        cnp.int64_t counts_sum, int_counts_sum
        cnp.int64_t int_counts_diff
        cnp.int32_t start,end,length
        Py_ssize_t i
        cnp.ndarray[cnp.int64_t, ndim=1] idata
        cnp.ndarray[cnp.int64_t, ndim=1] rems
        cnp.ndarray[cnp.float632_t, ndim=1] pvals

    for i in range(indptr.shape[0] - 1):
        start, end = indptr[i], indptr[i+1]
        length = end - start

        idata = data[start:end].astype(np.int64)
        counts_sum = idata.sum()

        # TODO: This could potentially be relaxed
        if counts_sum < n:
            data[start:end] = 0
            continue

        # get get int truncation of the fraction as the baseline
        data[start:end] = (n*idata) // counts_sum
        int_counts_sum = data[start:end].sum()

        # see how much off we are
        int_counts_diff = n-int_counts_sum
        if (int_counts_diff>0):
            # randomly pick elements from the remainders
            rems = (n*idata) % counts_sum
            rems_sum = rems.sum()
            pvals = rems.astype(np.float32) / rems_sum
            data[start:end] += rng.multinomial(int_counts_sum, pvals)


def _subsample(arr, n, with_replacement, rng, true_subsample):
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
    true_subsample : bool
        If `True`, use traditional subsampling. If `False`,
        use a faster approximate method.
        Ignored` if `with_replacement` is `True`.

    Returns
    -------
    ndarray
        Subsampled data

    Notes
    -----
    This code was adapted from scikit-bio (`skbio.math._subsample`)

    """
    if (with_replacement):
       return _subsample_with_replacement(arr.data, arr.indptr, n, rng)
    elif (true_subsample):
       return _subsample_without_replacement(arr.data, arr.indptr, n, rng)
    else:
       return _subsample_fast(arr.data, arr.indptr, n, rng)

