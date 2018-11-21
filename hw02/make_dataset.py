from multiprocessing.pool import ThreadPool

import numpy as np
from evaluaters import evaluate_func, evaluate_func_cheat


with open('./data/public.data', 'r') as file:
    x = np.empty([1000000, 10])
    for idx, line in enumerate(file):
        x[idx] = np.array([float(value) for value in line.strip().split(' ')])

    chunks = list()
    chunk_size = 10
    for idx in range(x.shape[0] // chunk_size):
        chunks.append(x[idx * chunk_size:(idx + 1) * chunk_size])

    # Make the Pool of workers
    pool = ThreadPool(8)

    # Open the urls in their own threads
    # and return the results
    results = np.concatenate(pool.map(evaluate_func_cheat, chunks))

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    print(results)
    with open('./data/public.data_x_y', 'w') as file_1:
        for idx in range(x.shape[0]):
            file_1.write(' '.join([str(value) for value in x[idx]]) + ' ' + str(results[idx]) + '\n')
