from RegressionObliviousTree import RegressionObliviousTree

from sklearn.tree import DecisionTreeRegressor

import numpy as np

from time import time

from utils import parce_sparce, test_sklearn_gbm

np.random.seed(142)

x_train, y_train = parce_sparce('Regression dataset/reg.train.txt', (7500, 246))
x_test, y_test = parce_sparce('Regression dataset/reg.test.txt', (10050, 246))


model = RegressionObliviousTree(fea_subsample=1.,
                                max_depth=10
                                )

st_time = time()
model.fit(x_train, y_train)
print(time() - st_time)

model_sk = DecisionTreeRegressor(max_depth=10)

st_time = time()
model_sk.fit(x_train, y_train)
print(time() - st_time)

print('Train MSE: ', np.mean((y_train - model.predict(x_train)) ** 2))
print('Test MSE: ', np.mean((y_test - model.predict(x_test)) ** 2))

print('Train MSE: ', np.mean((y_train - model_sk.predict(x_train)) ** 2))
print('Test MSE: ', np.mean((y_test - model_sk.predict(x_test)) ** 2))
