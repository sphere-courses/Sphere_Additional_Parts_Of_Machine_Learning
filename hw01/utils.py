import numpy as np


def parce_sparce(path, shape):
    x, y = np.zeros(shape), np.zeros([shape[0]])
    with open(path) as file:
        for idx, line in enumerate(file):
            y_x = line.strip().split(' ')
            y[idx], xs = y_x[0], y_x[1:]
            for seek, fea in (pair.split(':') for pair in xs):
                x[idx, int(seek)] = float(fea)
    return x, y
