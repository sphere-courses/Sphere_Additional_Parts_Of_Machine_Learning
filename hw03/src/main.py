import numpy as np
from sklearn.model_selection import train_test_split

from src.prepare_data import load_data_fm
from src.FM import FM


x, y = load_data_fm(True)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)

x_test = load_data_fm(False)

np.random.seed(45)
k_s = [
    2, 3, 4, 5
]
lw0_s = [
    5., 2., 2., 1.
]
lw_s = [
    5., 5., 6., 1.
]
lv_s = [
    5., 7., 8., 1.
]
n_s = [
    15, 10, 10, 10
]
s_s = [
    1., 0.8, 0.8, 0.9
]

models = [FM(
    n_iters=n_s[_],
    k=k_s[_], init_sigma=s_s[_],
    l_w0=lw0_s[_], l_w=lw_s[_], l_v=lv_s[_],
) for _ in range(3, 4)]

[model.fit(x_train, y_train, x_val, y_val) for model in models]
y_predicts = [model.predict(x_val) for model in models]
y_predict = np.sum(y_predicts, axis=0) / float(len(y_predicts))

rmse = np.sqrt(np.mean((y_val - y_predict) ** 2))
print('Validation score: ', rmse)


y_predicts = [model.predict(x_test) for model in models]
y_predict = np.sum(y_predicts, axis=0) / float(len(y_predicts))

y_predict[y_predict > 5.] = 5.
y_predict[y_predict < 1.] = 1.
with open('submission.txt', 'w') as file:
    file.write('Id,Mark\n')
    for idx, mark in enumerate(y_predict):
        file.write(str(idx + 1) + ',' + str(mark) + '\n')
