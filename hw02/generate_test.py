import math
import numpy as np
from evaluaters import evaluate_func_cheat

test_size = 1_000_000
x_test = np.random.uniform(0., 10., test_size * 10).reshape(test_size, 10)
y_test = evaluate_func_cheat(x_test)

with open('./data/test.data', 'w') as file:
    for idx in range(test_size):
        file.write(' '.join([str(round(value, 6)) for value in x_test[idx]]) + '\t' + str(y_test[idx]) + '\n')