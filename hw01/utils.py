import numpy as np
from sklearn.metrics import mean_squared_error


def parce_sparce(path, shape):
    x, y = np.zeros(shape), np.zeros([shape[0]])
    with open(path) as file:
        for idx, line in enumerate(file):
            y_x = line.strip().split(' ')
            y[idx], xs = y_x[0], y_x[1:]
            for seek, fea in (pair.split(':') for pair in xs):
                x[idx, int(seek)] = float(fea)
    return x, y


def test_sklearn_gbm(model, x_train, Y_train, x_test, Y_test, n_estimators_list):
    model.n_estimators = 1
    model.warm_start = True
    n_estimators_list = sorted(n_estimators_list)
    errors = []
    for est_num in n_estimators_list:
        model.n_estimators = est_num
        model.fit(x_train, Y_train)
        pred = model.predict(x_test)
        error = mean_squared_error(Y_test, pred)
        errors.append(error)
    return model.train_score_, errors
