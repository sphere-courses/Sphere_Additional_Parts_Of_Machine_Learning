import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

_so_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './../FM/FM/FM.so')
)
_lib = ctypes.cdll[os.path.join(os.getcwd(), _so_path)]


_predict_c = _lib['predict']
_predict_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double)
]
_predict_c.restype = ndpointer(dtype=ctypes.c_double, shape=(2, ))

_get_w_star_c = _lib['get_w_star']
_get_w_star_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32
]
_get_w_star_c.restype = ndpointer(dtype=ctypes.c_double, shape=(2, ))

_get_v_star_c = _lib['get_v_star']
_get_v_star_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32
]
_get_v_star_c.restype = ndpointer(dtype=ctypes.c_double, shape=(2, ))


class FM:
    def __init__(self, k=5, l_w0=1., l_w=1., l_v=1., init_sigma=0.5, n_iters=5):
        self.w0 = None
        self.w = None
        self.v = None

        self.k = k
        self.l_w0 = l_w0
        self.l_w = l_w
        self.l_v = l_v
        self.init_sigma = init_sigma
        self.n_iters = n_iters

    def predict(self, x):
        ptr_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_w = self.w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_v = self.v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_predict = np.empty([x.shape[0]], dtype=np.double)
        ptr_y_predict = y_predict.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        result = _predict_c(ptr_x, self.w0, ptr_w, ptr_v, x.shape[0], x.shape[1], self.v.shape[1], ptr_y_predict)
        return y_predict

    def _get_e(self, x, y):
        return y - self.predict(x)

    def _update_w0_star(self, x, e):
        n = float(x.shape[0])
        self.w0 = (n * self.w0 + np.sum(e)) / (n + self.l_w0)

    def _update_w_star(self, x, y, e):
        ptr_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_e = e.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_w = self.w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_v = self.v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        result = _get_w_star_c(ptr_x, ptr_y, ptr_e, self.w0, ptr_w, ptr_v, self.l_w, x.shape[0], x.shape[1], self.v.shape[1])

    def _update_v_star(self, x, y, e):
        ptr_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_e = e.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_w = self.w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ptr_v = self.v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        result = _get_v_star_c(ptr_x, ptr_y, ptr_e, self.w0, ptr_w, ptr_v, self.l_v, x.shape[0], x.shape[1], self.v.shape[1])

    def score(self, x, y):
        e = self._get_e(x, y)
        return np.sqrt(np.mean(e ** 2))

    def fit(self, x, y, x_test=None, y_test=None):
        self.w0 = np.random.normal(loc=0., scale=self.init_sigma, size=[1]).astype(np.double)
        self.w = np.random.normal(loc=0., scale=self.init_sigma, size=[x.shape[1]]).astype(np.double)
        self.v = np.random.normal(loc=0., scale=self.init_sigma, size=[x.shape[1], self.k]).astype(np.double)
        for _ in range(self.n_iters):
            e = self._get_e(x, y)
            self._update_w0_star(x, e)
            e = self._get_e(x, y)
            self._update_w_star(x, y, e)
            e = self._get_e(x, y)
            self._update_v_star(x, y, e)
            print("Iter: {0} Train score: {1}".format(str(_), str(self.score(x, y))))
            if x_test is not None:
                print("Iter: {0} Test score: {1}".format(str(_), str(self.score(x_test, y_test))))
