import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from utils import parce_sparce

from StochasticGradientBoosting import StochasticGradientBoosting


x_train, y_train = parce_sparce('Regression dataset/reg.train.txt', (7500, 246))
x_test, y_test = parce_sparce('Regression dataset/reg.test.txt', (10050, 246))

np.random.seed(10)

model = StochasticGradientBoosting(learning_rate=0.1,
                                   n_estimators=200,
                                   fea_subsample=0.05,
                                   max_depth=1,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   subsample=0.05
                                   )
mse_train, mse_test = model.fit(x_train, y_train, x_test, y_test)

print(mse_train)
print(mse_test)

print('Train MSE: ', np.mean((y_train - model.predict(x_train)) ** 2))
print('Test MSE: ', np.mean((y_test - model.predict(x_test)) ** 2))

plt.plot(mse_train, label='mse_train')
plt.plot(mse_test, label='mse_test')

mse_sk_train = []
mse_sk_test = []
for idx in range(1, 201):
    model_sk = GradientBoostingRegressor(learning_rate=0.1,
                                         n_estimators=idx,
                                         max_depth=1,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         subsample=.05
                                         )
    model_sk.fit(x_train, y_train)
    mse_sk_test.append(np.mean((y_test - model_sk.predict(x_test)) ** 2))

    print('idx: ', idx, model_sk.train_score_[-1], mse_sk_test[-1])
mse_sk_train = model_sk.train_score_

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
