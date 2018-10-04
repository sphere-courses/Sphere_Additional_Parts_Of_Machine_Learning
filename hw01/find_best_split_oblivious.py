import os

import ctypes

import numpy as np

from numpy.ctypeslib import ndpointer


_so_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './find_best_split/find_best_split/find_best_split.so')
)
_lib = ctypes.cdll[os.path.join(os.getcwd(), _so_path)]
_find_best_split_c = _lib['find_best_split']
_find_best_split_c.argtypes = [ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_int32,
                               ctypes.c_int32,
                               ctypes.c_int32,
                               ctypes.c_int32
                               ]
_find_best_split_c.restype = ndpointer(dtype=ctypes.c_double, shape=(2,))


def find_best_split(
        x_sliced_sorted, y_sliced_sorted,
        idx_sliced_sorted, node_idx,
        used_fea,
        depth):
    allowed_fea = np.ascontiguousarray(np.setxor1d(range(x_sliced_sorted.shape[0]), used_fea), dtype=np.int32)

    ptr_x_sliced_sorted = x_sliced_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_y_sliced_sorted = y_sliced_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_idx_sliced_sorted = idx_sliced_sorted.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    ptr_node_idx = node_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    ptr_allowed_fea = allowed_fea.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    split_fea, split_thr = _find_best_split_c(ptr_x_sliced_sorted, ptr_y_sliced_sorted,
                                              ptr_idx_sliced_sorted, ptr_node_idx,
                                              ptr_allowed_fea, len(allowed_fea),
                                              x_sliced_sorted.shape[1], y_sliced_sorted.shape[0], depth)

    return int(round(split_fea)), split_thr
