import numpy as np


class _Node:
    def __init__(self, node_idx, eps=1e-4):
        self.node_idx = node_idx
        self.eps = eps
        self.is_leaf = False
        self.depth = None

        self.n_obj_train = None
        self.n_obj_last_predict = None

        self.n_best_fea = None
        self.threshold = None

        self.left_node = None
        self.right_node = None

        self.prediction = None

    def fit(self, x, y, depth, max_depth, min_samples_split=2, min_samples_leaf=1):
        self.fit_(x, y, depth, max_depth, min_samples_split, min_samples_leaf)

    def fit_(self, x, y, depth, max_depth, min_samples_split=2, min_samples_leaf=1):
        self.n_obj_train = x.shape[0]
        self.depth = depth

        if depth == max_depth or x.shape[0] < min_samples_split:
            self.is_leaf = True
            self.prediction = np.mean(y)
            return

        x_sliced = x
        y_sliced = y

        idx_sliced_sorted = np.argsort(x_sliced, axis=0)

        x_sliced_sorted = np.sort(x_sliced, axis=0)
        y_sliced_sorted = y_sliced[idx_sliced_sorted]

        y_sq_cumsum = np.cumsum(y_sliced_sorted ** 2, axis=0)
        y_sq_cumsum_reversed = np.cumsum(y_sliced_sorted[::-1] ** 2, axis=0)[::-1]

        y_cumsum = np.cumsum(y_sliced_sorted, axis=0)
        y_cumsum_reversed = np.cumsum(y_sliced_sorted[::-1], axis=0)[::-1]

        cnt_array = np.array(range(1, y_cumsum.shape[0] + 1)).reshape(-1, 1)
        mse_left = y_sq_cumsum - y_cumsum ** 2 / cnt_array
        mse_right = y_sq_cumsum_reversed - y_cumsum_reversed ** 2 / cnt_array[::-1]

        mse = mse_left
        mse[:-1] += mse_right[1:]

        # exclude non valuable slices
        garbage_fea = np.where(np.abs(np.max(x, axis=0) - np.min(x, axis=0)) < self.eps)[0]
        if garbage_fea.shape[0] > 0:
            mse[:, garbage_fea] = np.nan
        # exclude incorrect border slices
        mse[np.isclose(x_sliced_sorted, np.roll(x_sliced_sorted, shift=-1, axis=0))] = np.nan

        try:
            n_best_obj, self.n_best_fea = np.unravel_index(np.nanargmin(mse), mse.shape)
        except ValueError:
            self.is_leaf = True
            self.prediction = np.mean(y)
            return

        self.threshold = x_sliced_sorted[n_best_obj, self.n_best_fea]

        left_obj = np.where(x[:, self.n_best_fea] <= self.threshold)[0]
        right_obj = np.where(x[:, self.n_best_fea] > self.threshold)[0]
        if left_obj.shape[0] < min_samples_leaf or right_obj.shape[0] < min_samples_leaf:
            self.is_leaf = True
            self.prediction = np.mean(y)
            return

        self.left_node = _Node(2 * self.node_idx + 1)
        self.right_node = _Node(2 * self.node_idx + 2)
        self.left_node.fit(x[left_obj], y[left_obj], depth + 1, max_depth, min_samples_split)
        self.right_node.fit(x[right_obj], y[right_obj], depth + 1, max_depth, min_samples_split)

    def predict(self, x):
        self.n_obj_last_predict = x.shape[0]
        if self.is_leaf:
            if self.depth > 0:
                return self.prediction
            return np.repeat(np.array([self.prediction]), x.shape[0])

        prediction = np.empty([x.shape[0]])
        left_obj = np.where(x[:, self.n_best_fea] <= self.threshold)
        right_obj = np.where(x[:, self.n_best_fea] > self.threshold)

        prediction[left_obj] = self.left_node.predict(x[left_obj])
        prediction[right_obj] = self.right_node.predict(x[right_obj])

        return prediction

    def traverse(self):
        if self.is_leaf:
            return
        self.left_node.traverse()
        self.right_node.traverse()

    def update_leafs(self, learning_rate, gamma):
        if self.is_leaf:
            self.prediction *= learning_rate * gamma
            return
        self.left_node.update_leafs(learning_rate, gamma)
        self.right_node.update_leafs(learning_rate, gamma)


class RegressionDecisionTree:
    def __init__(self, fea_subsample=1., max_depth=1, min_samples_split=2, min_samples_leaf=1):
        self.fea_subsample = fea_subsample
        self.fea_idx = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.top_node = _Node(node_idx=0)

    def fit(self, x, y):
        self.fea_idx = np.random.choice(x.shape[1], int(x.shape[1] * self.fea_subsample), replace=False)
        self.top_node.fit(x[:, self.fea_idx], y, 0, self.max_depth, self.min_samples_split, self.min_samples_leaf)
        return self

    def predict(self, x):
        return self.top_node.predict(x[:, self.fea_idx])

    def traverse(self):
        self.top_node.traverse()

    def update_leafs(self, learning_rate, gamma):
        self.top_node.update_leafs(learning_rate, gamma)
