from libcpp.map cimport map
cimport cython
import logging
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_values(array_np, values_map):

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = array_np.size()

    replaced_np = np.zeros_like(array_np)
    cdef np.npy_uint64[:] array = array_np
    cdef np.npy_uint64[:] replaced = replaced_np
    cdef map[np.npy_uint64, np.npy_uint64] cmap = values_map

    for i in range(n):
        replaced[i] = cmap[array[i]]

    return replaced_np
