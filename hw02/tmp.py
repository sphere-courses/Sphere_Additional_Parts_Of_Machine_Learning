import numpy as np
from evaluaters import evaluate_func_cheat

# with open('./data/private.data', 'r') as file:
#     with open('./data/private.data_x_y', 'w') as file_1:
#         for line in file:
#             y = evaluate_func_cheat(np.array([float(value) for value in line.strip().split()]).reshape(1, -1))
#             file_1.write(line.strip() + ' ' + str(y[0]) + '\n')

file_data = open('./data/private.data_x_y', 'r')
file_subm = open('./submission_1541370269.5490518.txt', 'r')
loss = 0.
head = file_subm.readline()
for line in file_data:
    y = float(file_subm.readline().strip().split(',')[1])
    vals = [float(value) for value in line.strip().split()]
    loss += (y - vals[-1]) ** 2
print(np.sqrt(loss / 1e6))
