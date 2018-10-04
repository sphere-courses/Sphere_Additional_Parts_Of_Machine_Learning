import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from utils import parce_sparce, test_sklearn_gbm

from StochasticGradientBoosting import StochasticGradientBoosting


x_train, y_train = parce_sparce('Regression dataset/reg.train.txt', (7500, 246))
x_test, y_test = parce_sparce('Regression dataset/reg.test.txt', (10050, 246))

# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# boston = load_boston()
#
# x = boston['data']
# y = boston['target']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

np.random.seed(10)


model = StochasticGradientBoosting(learning_rate=0.1,
                                   n_estimators=300,
                                   fea_subsample=1.,
                                   max_depth=3,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   subsample=1.
                                   )
mse_train, mse_test = model.fit(x_train, y_train, x_test, y_test)

print(mse_train)
print(mse_test)

print('Train MSE: ', np.mean((y_train - model.predict(x_train)) ** 2))
print('Test MSE: ', np.mean((y_test - model.predict(x_test)) ** 2))

plt.plot(mse_train, label='mse_train')
plt.plot(mse_test, label='mse_test')


model_sk = GradientBoostingRegressor(learning_rate=0.1,
                                     n_estimators=300,
                                     max_depth=3,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     subsample=1.
                                     )

mse_sk_train, mse_sk_test = test_sklearn_gbm(model_sk, x_train, y_train, x_test, y_test, range(1, 300))

print(mse_sk_train)
print(mse_sk_test)

print('Train MSE: ', mse_sk_train[-1])
print('Test MSE: ', mse_sk_test[-1])

plt.plot([x * 1.03 for x in mse_sk_train], linestyle='--', linewidth=0.2)
plt.plot([x * 0.97 for x in mse_sk_train], linestyle='--', linewidth=0.2)
plt.plot(mse_sk_train, label='mse_sk_train')
plt.plot([x * 1.03 for x in mse_sk_test], linestyle='--', linewidth=0.2)
plt.plot([x * 0.97 for x in mse_sk_test], linestyle='--', linewidth=0.2)
plt.plot(mse_sk_test, label='mse_sk_test')

plt.savefig('plot.png')
