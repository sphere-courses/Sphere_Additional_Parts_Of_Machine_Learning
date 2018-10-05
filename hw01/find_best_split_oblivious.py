import os

from math import isclose

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


def find_best_split_python(
        x_sliced_sorted, y_sliced_sorted,
        idx_sliced_sorted, node_idx,
        used_fea,
        depth):
    allowed_fea = np.setxor1d(range(x_sliced_sorted.shape[0]), used_fea).astype(np.int32)

    n_obj, n_fea = x_sliced_sorted.shape[1], y_sliced_sorted.shape[0]
    node_bias = int((1 << depth) - 1)
    n_node = int((1 << depth))

    S_n_1 = np.zeros([n_node], dtype=np.double)
    S_p = np.zeros([n_node], dtype=np.double)
    n_n_1 = np.zeros([n_node], dtype=np.int32)
    n_p = np.zeros([n_node], dtype=np.int32)

    for i in range(n_obj):
        S_n_1[node_idx[idx_sliced_sorted[1, i]] - node_bias] += y_sliced_sorted[1, i]
        n_n_1[node_idx[idx_sliced_sorted[1, i]] - node_bias] += 1

    best_fea = -1
    best_antiloss = 0.
    best_thr = 0.
    for r in range(len(allowed_fea)):
        k = allowed_fea[r]

        S_p.fill(0.)
        n_p.fill(0)

        for i in range(n_obj - 1):
            S_p[node_idx[idx_sliced_sorted[k, i]] - node_bias] += y_sliced_sorted[k, i]
            n_p[node_idx[idx_sliced_sorted[k, i]] - node_bias] += 1
            antiloss = evaluate_antiloss(S_p, n_p, S_n_1, n_n_1, n_node)

            if not isclose(x_sliced_sorted[k, i], x_sliced_sorted[k, i + 1]) and antiloss > best_antiloss:
                best_antiloss = antiloss
                best_fea = k
                best_thr = x_sliced_sorted[k, i]

    return best_fea, best_thr


def evaluate_antiloss(S_p, n_p, S_n_1, n_n_1, n_node):
    antiloss = 0.
    for j in range(n_node):
        antiloss += (S_p[j] * S_p[j] / n_p[j] if n_p[j] else 0.) + \
                    ((S_n_1[j] - S_p[j]) * (S_n_1[j] - S_p[j]) / (n_n_1[j] - n_p[j]) if (n_n_1[j] - n_p[j]) else 0.)
    return antiloss
