import time

import numpy as np


def extract_features(values):
    result = np.empty([values.shape[0], 65])
    ptr = values.shape[1]
    result[:, :values.shape[1]] = values
    for idx in range(values.shape[1]):
        for jdx in range(values.shape[1]):
            if jdx > idx:
                break
            result[:, ptr] = values[:, idx] * values[:, jdx]
            ptr += 1
    return result


def sample_around(point, n_samples, eps):
    shift = np.random.uniform(-eps, eps, [n_samples, point.shape[1]])
    result = point + shift
    result *= np.sign(result)
    result[np.isclose(result, 0.)] = 1e-1
    return result


def get_data(path='./data/public.data_x_y'):
    with open(path, 'r') as file:
        x = np.empty([1000000, 10], dtype=np.float64)
        y = np.empty([1000000], dtype=np.float64)
        for idx, line in enumerate(file):
            values = [float(value) for value in line.strip().split()]
            x[idx], y[idx] = np.array(values[:10]), values[-1]
    return x, y


def validate(model, x=None, y=None):
    if x is None:
        x, y = get_data()
    x_0 = x[np.any(x < 1e-1, axis=1)]
    y_0 = y[np.any(x < 1e-1, axis=1)]
    x_1 = x[np.all(x > 1e-1, axis=1)]
    y_1 = y[np.all(x > 1e-1, axis=1)]
    loss = np.sum((model[0].predict(x_0) - y_0) ** 2) + np.sum((model[1].predict(x_1) - y_1) ** 2)
    return np.sqrt(loss / x.shape[0])


def make_submission(model, x):
    with open('submission_' + str(time.time()) + '.txt', 'w') as file:
        y = np.empty([x.shape[0]])
        y[np.any(x < 1e-1, axis=1).reshape(-1)] = model[0].predict(x[np.any(x < 1e-1, axis=1).reshape(-1)])
        y[np.all(x > 1e-1, axis=1).reshape(-1)] = model[1].predict(x[np.all(x > 1e-1, axis=1).reshape(-1)])
        file.write('Id,Expected\n')
        for idx in range(y.shape[0]):
            file.write(str(idx + 1) + ',' + str(y[idx]) + '\n')

