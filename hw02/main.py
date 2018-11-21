import os
import pickle

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from model import Ensemble
from evaluaters import evaluate_func_cheat as evaluate_func
from utils import get_data, validate, extract_features, make_submission, sample_around


# number of support dots
n_support = 10
# number of dots to choose the most uncertain from
first_sample_size = 10_000
# number of the most uncertain dots to add to dataset
second_sample_size = 5
# number of train iterates
n_iters = 5000

model_0 = Ensemble(DecisionTreeRegressor, n_models=4)
model_1 = Ensemble(DecisionTreeRegressor, n_models=4)

# y_train is used only for validation, not for model training
x_train, y_train = get_data()
x_test, y_test = get_data('./data/private.data_x_y')
# fea_train = extract_features(x_train)
# fea_test = extract_features(x_test)
fea_train = x_train
fea_test = x_test

prefix = 'no_fea_do_not_use_eps'

best_full_rmse = 500
best_test_rmse = 1350

idx_chosen = np.random.randint(0, x_train.shape[0], [n_support])
y_chosen = np.array(evaluate_func(x_train[idx_chosen]))
# for path in os.listdir('./checkpoints/'):
#     if os.path.exists('./checkpoints/' + path):
#         idx_chosen = pickle.load(open('./checkpoints/' + path, 'rb'))
#     else:
#         idx_chosen = np.random.randint(0, x_train.shape[0], [n_support])
#     y_chosen = np.array(evaluate_func(x_train[idx_chosen]))
#
#     f_0 = fea_train[idx_chosen][np.any(fea_train[idx_chosen] < 1e-1, axis=1)]
#     y_0 = y_chosen[np.any(fea_train[idx_chosen] < 1e-1, axis=1)]
#
#     f_1 = fea_train[idx_chosen][np.all(fea_train[idx_chosen] > 1e-1, axis=1)]
#     y_1 = y_chosen[np.all(fea_train[idx_chosen] > 1e-1, axis=1)]
#
#     model_0.fit(f_0, y_0)
#     model_1.fit(f_1, y_1)
#
#     full_rmse = int(validate([model_0, model_1], fea_train, y_train))
#     test_rmse = int(validate([model_0, model_1], fea_test, y_test))
#     print('Full RMSE: ', full_rmse)
#     print('Test RMSE: ', test_rmse)
#     if test_rmse <= 1850:
#         make_submission([model_0, model_1], fea_test)
#         quit(0)

for n_iter in range(n_iters):
    f_0 = fea_train[idx_chosen][np.any(fea_train[idx_chosen] < 1e-1, axis=1)]
    y_0 = y_chosen[np.any(fea_train[idx_chosen] < 1e-1, axis=1)]

    f_1 = fea_train[idx_chosen][np.all(fea_train[idx_chosen] > 1e-1, axis=1)]
    y_1 = y_chosen[np.all(fea_train[idx_chosen] > 1e-1, axis=1)]

    model_0.fit(f_0, y_0)
    model_1.fit(f_1, y_1)

    print('Iter: ', n_iter)
    full_rmse = int(validate([model_0, model_1], fea_train, y_train))
    test_rmse = int(validate([model_0, model_1], fea_test, y_test))
    print('Full RMSE: ', full_rmse)
    print('Test RMSE: ', test_rmse)
    print('Partial RMSE: ', validate([model_0, model_1], fea_train[idx_chosen], y_train[idx_chosen]))

    # if full_rmse < best_full_rmse:
    #     pickle.dump(idx_chosen, open('./checkpoint_' + prefix + '_' + str(model.base_model) + '_' + str(full_rmse) + '_' + str(test_rmse) + '.pkz', 'wb'))
    #     best_full_rmse = full_rmse
    #
    # if test_rmse < best_test_rmse:
    #     pickle.dump(idx_chosen, open('./checkpoint_' + prefix + '_' + str(model.base_model) + '_' + str(full_rmse) + '_' + str(test_rmse) + '.pkz', 'wb'))
    #     best_test_rmse = test_rmse

    if np.random.uniform(0., 1., 1) < 0.8:
        idx_first_sample = np.random.randint(0, x_train.shape[0], [first_sample_size])
        variances_0 = np.var(model_0.predict_all_models(fea_train[idx_first_sample]), axis=1)
        variances_1 = np.var(model_1.predict_all_models(fea_train[idx_first_sample]), axis=1)
        idx_chosen_new_0 = idx_first_sample[np.argsort(-variances_0)[:second_sample_size]]
        idx_chosen_new_1 = idx_first_sample[np.argsort(-variances_1)[:second_sample_size]]
        idx_chosen_new = np.concatenate([idx_chosen_new_0, idx_chosen_new_1])
    else:
        idx_chosen_new = np.random.randint(0, x_train.shape[0], [second_sample_size])

    idx_chosen = np.concatenate([idx_chosen, idx_chosen_new])
    y_chosen = np.concatenate([y_chosen, evaluate_func(x_train[idx_chosen_new])])
pickle.dump(idx_chosen, open('./checkpoint.pkz', 'wb'))

