from collections import defaultdict

import numpy as np


def load_data_fm(train=True):
    if train:
        path = './../data/train.txt'
        shape = [90570, 943 + 1682 * 2]
    else:
        path = './../data/test.txt'
        shape = [9430, 943 + 1682 * 2]
    with open(path, 'r') as file:
        x = np.zeros(shape, dtype=np.float64)
        y = np.zeros(shape[0], dtype=np.float64)
        for idx, line in enumerate(file):
            if train:
                user_id, film_id, mark = [int(val) for val in line.strip().split()]
                y[idx] = mark
            else:
                user_id, film_id = [int(val) for val in line.strip().split()]
            x[idx][user_id - 1] = 1
            x[idx][943 + film_id - 1] = 1
        if train:
            return x, y
        else:
            return x


def load_data_fm_svd(train=True):
    if train:
        path = './../data/train.txt'
        shape = [90570, 943 + 1682 * 2]
    else:
        path = './../data/test.txt'
        shape = [9430, 943 + 1682 * 2]
    user_movies = defaultdict(set)

    with open('./../data/train.txt', 'r') as file:
        for idx, line in enumerate(file):
            user_id, film_id, _ = [int(val) for val in line.strip().split()]
            user_movies[user_id].add(film_id)

    with open('./../data/test.txt', 'r') as file:
        for idx, line in enumerate(file):
            user_id, film_id = [int(val) for val in line.strip().split()]
            user_movies[user_id].add(film_id)

    with open(path, 'r') as file:
        x = np.zeros(shape, dtype=np.float64)
        y = np.zeros(shape[0], dtype=np.float64)
        for idx, line in enumerate(file):
            if train:
                user_id, film_id, mark = [int(val) for val in line.strip().split()]
                y[idx] = mark
            else:
                user_id, film_id = [int(val) for val in line.strip().split()]
            x[idx][user_id - 1] = 1
            x[idx][943 + film_id - 1] = 1
            Nu = len(user_movies[user_id])
            fea = 0 if Nu == 0 else 1. / np.sqrt(Nu)
            for rated_film_id in user_movies[user_id]:
                x[idx][943 + 1682 + rated_film_id - 1] = fea
        if train:
            return x, y
        else:
            return x
