import numpy as np


class Ensemble:
    def __init__(self, base_model, n_models, *args, **kwargs):
        self.base_model = base_model
        self.n_models = n_models
        self.models = [self.base_model(*args, **kwargs) for _ in range(self.n_models)]

    def predict_all_models(self, x):
        return np.hstack([model.predict(x).reshape(-1, 1) for model in self.models])

    def predict(self, x):
        if x.shape[0] == 0:
            return 0.
        return np.mean(self.predict_all_models(x), axis=1)

    def fit(self, x, y):
        for model in self.models:
            if x.shape[0] > 1:
                idxs = np.random.randint(0, x.shape[0], [int(x.shape[0] * 0.9)])
                model.fit(x[idxs], y[idxs])
            elif x.shape[0] == 1:
                model.fit(x, y)
            else:
                model.fit(
                    np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]),
                    np.array([97.404])
                )