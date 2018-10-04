import numpy as np

# from RegressionDecisionTree import RegressionDecisionTree

from RegressionObliviousTree import RegressionObliviousTree


class StochasticGradientBoosting:
    def __init__(self,
                 learning_rate,
                 n_estimators,
                 fea_subsample,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 subsample
                 ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.fea_subsample = fea_subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

        self.estimators = []
        self.b = np.ones([self.n_estimators])

    def fit(self, x, y, x_test=None, y_test=None):
        mse_train = []
        mse_test = []

        self.estimators.append(RegressionObliviousTree(fea_subsample=self.fea_subsample,
                                                       max_depth=0,
                                                       min_samples_split=self.min_samples_split,
                                                       min_samples_leaf=self.min_samples_leaf
                                                       ).fit(x, y)
                               )

        predictions_base = self.estimators[-1].predict(x)

        if x_test is not None:
            mse_train.append(np.mean((predictions_base - y) ** 2))
            predictions_base_test = np.sum((tree.predict(x_test) for tree in self.estimators), axis=1)
            mse_test.append(np.mean((predictions_base_test - y_test) ** 2))

        for idx in range(self.n_estimators - 1):
            print('idx: ', idx, mse_train[-1], mse_test[-1])
            indices = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=False)
            gradients = 2.0 * (y[indices] - predictions_base[indices])

            self.estimators.append(RegressionObliviousTree(fea_subsample=self.fea_subsample,
                                                           max_depth=self.max_depth,
                                                           min_samples_split=self.min_samples_split,
                                                           min_samples_leaf=self.min_samples_leaf
                                                           ).fit(x[indices], gradients)
                                   )

            prediction_i = self.estimators[-1].predict(x)
            betta = ((prediction_i[indices] * (y[indices] - predictions_base[indices])).sum() /
                     ((prediction_i[indices]**2).sum()))
            self.estimators[-1].update_leafs(self.learning_rate, betta)
            self.estimators[-1].scale_leafs(type='no_k')
            predictions_base += self.estimators[-1].predict(x)

            if x_test is not None:
                mse_train.append(np.mean((predictions_base - y)**2))
                predictions_base_test += self.estimators[-1].predict(x_test)
                mse_test.append(np.mean((predictions_base_test - y_test)**2))
        if x_test is not None:
            return mse_train, mse_test

    def predict(self, x):
        return np.sum((tree.predict(x) for tree in self.estimators))
