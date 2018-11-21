import subprocess
import numpy as np


def evaluate_func(values, path='AL_Oracle/2018/home/vgulin/code/active_learning/Oracle'):
    if values.shape[1] != 10:
        raise RuntimeError
    result = np.empty([values.shape[0]])
    for idx in range(values.shape[0]):
        args = ' '.join(str(value) for value in values[idx])
        result[idx] = subprocess.check_output(path + ' ' + args, shell=True)
    return result


def evaluate_func_cheat(values):
    if values.shape[1] != 10:
        raise RuntimeError
    return (
            values[:, 1] * values[:, 6] +
            values[:, 8] / values[:, 9] * np.sqrt(values[:, 6] / values[:, 7]) +
            np.pi * np.sqrt(values[:, 2]) +
            1. / np.sin(values[:, 3]) +
            np.log(values[:, 2] + values[:, 4])
    )
