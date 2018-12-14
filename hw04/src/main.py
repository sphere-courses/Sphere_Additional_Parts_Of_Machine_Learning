import os
import zlib
import pickle
import numpy as np
from xgboost import XGBClassifier
from collections import defaultdict
from catboost import CatBoostClassifier


'''
timestamp\  0

    ;label\ 1
    --------------
    ;C1\    2
    ;C2\    3
    ;C3\    4
    ;C4\    5
    ;C5\    6
    ;C6\    7
    ;C7\    8
    ;C8\    9
    ;C9\    10
    ;C10\   11
    --------------
    ;CG1\   12
    ;CG2\   13
    ;CG3\   14
    --------------
    ;l1\    15
    ;l2\    16
    --------------
    ;C11\   17
    ;C12    18
'''
idx_timestamp = 0
idx_label = 1
idx_cat = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18]
idx_grp = [12, 13, 14]
idx_l = [15, 16]


def split(line):
    objects = line.strip().split(';')
    print('label: ' + objects[idx_label])
    for idx in idx_cat:
        print('categorical: ' + objects[idx])
    for idx in idx_grp:
        print('group: ' + objects[idx])
    for idx in idx_l:
        print('counters: ' + objects[idx])


def get_vw_line(objects, test=False, statistics=None, use_statistics=False, use_online_statistics=False):
    result = ''
    if not test:
        if objects[idx_label] == '0':
            result += '-1 '
        else:
            result += '1 '

    # integer features
    result += '|L '
    for _, idx in enumerate(idx_l):
        result += '{0:d}:'.format(_ + 1) + objects[idx] + ' '

    # categorical features
    result += '|C '
    for _, idx in enumerate(idx_cat):
        result += objects[idx] + ' '

    # categorical counters
    result += '|T '
    for _, idx in enumerate(idx_cat):
        fea_name = 'C{0:d}_'.format(_ + 1) + objects[idx]
        if not test and not use_statistics or use_online_statistics:
            statistics[0][fea_name] += 1
            if objects[idx_label] == '1':
                statistics[1][fea_name] += 1
        if use_statistics:
            # result += '{0:d}_1:{1:d}'.format(_ + 1, statistics[0][fea_name]) + ' '
            # result += '{0:d}_2:{1:d}'.format(_ + 1, statistics[1][fea_name]) + ' '
            result += '{0:d}_3:{1:.4f}'.format(_ + 1, 1000. * statistics[1][fea_name] / (statistics[0][fea_name] + 1e-5)) + ' '

    # group features
    for _, idx in enumerate(idx_grp):
        result += '|G{0:d} '.format(_ + 1)

        # add permutation ignorant hashing
        sub_objects = sorted(objects[idx].split(','))
        hash_sum = sum(list(map(int, sub_objects))) if len(sub_objects) > 1 or sub_objects[0] != '' else 0
        result += str(hash_sum) + ' '

        result += ' '.join(sub_objects)

    # group counters
    for _, idx in enumerate(idx_grp):
        result += '|P{0:d} '.format(_ + 1)

        # add permutation ignorant hashing
        sub_objects = sorted(objects[idx].split(','))
        hash_sum = sum(list(map(int, sub_objects))) if len(sub_objects) > 1 or sub_objects[0] != '' else 0

        fea_name = 'G{0:d}_{1:d}'.format(_ + 1, hash_sum)
        if not test and not use_statistics or use_online_statistics:
            statistics[0][fea_name] += 1
            if objects[idx_label] == '1':
                statistics[1][fea_name] += 1
        if use_statistics:
            # result += '{0:d}_1:{1:d}'.format(_ + 1, statistics[0][fea_name]) + ' '
            # result += '{0:d}_2:{1:d}'.format(_ + 1, statistics[1][fea_name]) + ' '
            result += '{0:d}_3:{1:.4f}'.format(_ + 1, 1000. * statistics[1][fea_name] / (statistics[0][fea_name] + 1e-5)) + ' '

    return result


def get_ctr_line(objects, test=False, statistics=None, use_statistics=False):
    result = ''
    if not test:
        if objects[idx_label] == '0':
            result += '-1 '
        else:
            result += '1 '

    # integer features
    for _, idx in enumerate(idx_l):
        result += objects[idx] + ' '

    # categorical counters
    for _, idx in enumerate(idx_cat):
        fea_name = 'C{0:d}_'.format(_ + 1) + objects[idx]
        if not test and not use_statistics:
            statistics[0][fea_name] += 1
            if objects[idx_label] == '1':
                statistics[1][fea_name] += 1
        if use_statistics:
            # result += '{0:d}'.format(statistics[0][fea_name]) + ' '
            # result += '{0:d}'.format(statistics[1][fea_name]) + ' '
            result += '{0:.4f}'.format(1000. * statistics[1][fea_name] / (statistics[0][fea_name] + 1e-5)) + ' '

    # group counters
    for _, idx in enumerate(idx_grp):
        # add permutation ignorant hashing
        sub_objects = sorted(objects[idx].split(','))
        hash_sum = sum(list(map(int, sub_objects))) if len(sub_objects) > 1 or sub_objects[0] != '' else 0

        fea_name = 'G{0:d}_{1:d}'.format(_ + 1, hash_sum)
        if not test and not use_statistics:
            statistics[0][fea_name] += 1
            if objects[idx_label] == '1':
                statistics[1][fea_name] += 1
        if use_statistics:
            # result += '{0:d}'.format(statistics[0][fea_name]) + ' '
            # result += '{0:d}'.format(statistics[1][fea_name]) + ' '
            result += '{0:.4f}'.format(1000. * statistics[1][fea_name] / (statistics[0][fea_name] + 1e-5)) + ' '

    return result


def convert_to_vw_format(path, test=False, statistics=None, use_statistics=False, use_online_statistics=False):
    with open(path, 'r') as input_file:
        with open(path + '.vw', 'w') as output_file:
            # skip header
            header = input_file.readline()
            for line in input_file:
                objects = line.strip().split(';')
                output_file.write(get_vw_line(objects, test, statistics, use_statistics, use_online_statistics) + '\n')


def convert_to_ctr_format(path, test=False, statistics=None, use_statistics=False):
    with open(path, 'r') as input_file:
        with open(path + '.ctr', 'w') as output_file:
            # skip header
            header = input_file.readline()
            for line in input_file:
                objects = line.strip().split(';')
                output_file.write(get_ctr_line(objects, test, statistics, use_statistics) + '\n')


def convert_to_submition(path, probas=None):
    with open(path + '.submition', 'w') as output_file:
        output_file.write('Id,Click\n')
        if probas is not None:
            for idx, prediction in enumerate(probas):
                output_file.write('{0:d},{1:.5f}\n'.format(idx + 1, prediction[1]))
        else:
            with open(path, 'r') as input_file:
                for idx, line in enumerate(input_file):
                    probas = list(map(float, line.strip().split()))
                    prediction = sum(probas) / len(probas)
                    output_file.write('{0:d},{1:.5f}\n'.format(idx + 1, prediction))


def get_data(path, test=False):
    if test:
        x = np.zeros([20317220, 17], dtype=np.float32)
        with open(path, 'r') as file:
            for idx, line in enumerate(file):
                objects = line.strip().split(' ')
                x[idx] = list(map(float, objects))
                if idx % 1_000_000 == 0:
                    print(idx)
        return x
    else:
        x, y = np.zeros([29989752, 17], dtype=np.float32), np.zeros([29989752, 1], dtype=np.int)
        with open(path, 'r') as file:
            for idx, line in enumerate(file):
                objects = line.strip().split(' ')
                x[idx] = list(map(float, objects[1:]))
                y[idx] = int(objects[0])
                if idx % 1_000_000 == 0:
                    print(idx)
        return x, y


# Use global statistics
# extract statistics
# appeared, clicked = defaultdict(int), defaultdict(int)
# convert_to_vw_format('/home/nahodnov17/hw04/data/train.csv', False, [appeared, clicked])
# with open('/home/nahodnov17/hw04/data/statistics.pkz', 'wb') as file:
#     pickle.dump([appeared, clicked], file)

# convert to vw format
# with open('/home/nahodnov17/hw04/data/statistics.pkz', 'rb') as file:
#     appeared, clicked = pickle.load(file)
# convert_to_vw_format('/home/nahodnov17/hw04/data/test.csv', True, [appeared, clicked], False)
# convert_to_vw_format('/home/nahodnov17/hw04/data/train.csv', False, [appeared, clicked], False)


# Use local statistics
# appeared, clicked = defaultdict(int), defaultdict(int)
# convert_to_vw_format('/home/nahodnov17/hw04/data/train.csv', False, [appeared, clicked], True, True)
# with open('/home/nahodnov17/hw04/data/statistics.pkz', 'wb') as file:
#     pickle.dump([appeared, clicked], file)
# with open('/home/nahodnov17/hw04/data/statistics.pkz', 'rb') as file:
#     appeared, clicked = pickle.load(file)
# convert_to_vw_format('/home/nahodnov17/hw04/data/test.csv', True, [appeared, clicked], True, False)

# convert to ctrs
# with open('/home/nahodnov17/hw04/data/statistics.pkz', 'rb') as file:
#     appeared, clicked = pickle.load(file)
# convert_to_ctr_format('/home/nahodnov17/hw04/data/test.csv', True, [appeared, clicked], True)
# convert_to_ctr_format('/home/nahodnov17/hw04/data/train.csv', False, [appeared, clicked], True)


# fit model
# "vw train.csv.vw --loss_function logistic -b 27 --passes 6 --holdout_off -C -0.4878 --l2 2e-8 -l 0.01 --adaptive --invariant --power_t 0.25 -c --compressed -f model.vw -k -ftrl"
# predict
# "vw -d test.csv.vw -t -i model.vw -p probas.txt --loss_function logistic --link logistic"

# convert file with probas to submition
convert_to_submition('/home/nahodnov17/hw04/data/probas.txt')


# load and pickle ctr data
# x, y = get_data('/home/nahodnov17/hw04/data/train.csv.ctr', test=False)
# x_validate = get_data('/home/nahodnov17/hw04/data/test.csv.ctr', test=True)
# with open('/home/nahodnov17/hw04/data/ctr.pkz', 'wb') as file:
#     pickle.dump([x, y, x_validate], file)

# with open('/home/nahodnov17/hw04/data/ctr.pkz', 'rb') as file:
#     x, y, x_validate = pickle.load(file)

# x_train, x_test, y_train, y_test = x[:20_000_000], x[20_000_000:], y[:20_000_000], y[20_000_000:]
# x_train, x_test, y_train, y_test = x[:1_000_000], x[1_000_000:2_000_000], y[:1_000_000], y[1_000_000:2_000_000]

# CatBoost on ctr data
# model = CatBoostClassifier(n_estimators=20, silent=False, thread_count=16, max_depth=2, loss_function='Logloss')
# model.fit(x_test, y_test, eval_set=[(x_test, y_test)], verbose_eval=True)
# probas = model.predict_proba(x_validate, thread_count=16)
# convert_to_submition('/home/nahodnov17/hw04/data/probas.txt', probas)

# XGBoost on ctr data
# model = XGBClassifier(n_estimators=10, silent=False)
# model.fit(x_train, y_train, eval_metric="logloss", eval_set=[(x_test, y_test)], verbose=True)
# probas = model.predict_proba(x_validate)
