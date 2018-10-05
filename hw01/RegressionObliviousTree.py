import numpy as np

from find_best_split_oblivious import find_best_split, find_best_split_python


class RegressionObliviousTree:
    def __init__(self, fea_subsample=1., max_depth=1, min_samples_split=2, min_samples_leaf=1):
        self.fea_subsample = fea_subsample
        self.fea_idx = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.split_fea = None
        self.split_thr = None
        self.node_pred = None
        self.n_obj_train = None

    def fit(self, x, y):
        self.fea_idx = np.random.choice(x.shape[1], int(x.shape[1] * self.fea_subsample), replace=False)

        x_sliced = np.transpose(x[:, self.fea_idx])
        y_sliced = np.transpose(y)

        idx_sliced_sorted = np.ascontiguousarray(np.argsort(x_sliced, axis=1), dtype=np.int32)

        x_sliced_sorted = np.ascontiguousarray(np.sort(x_sliced, axis=1), dtype=np.float64)
        y_sliced_sorted = np.ascontiguousarray(y_sliced[idx_sliced_sorted], dtype=np.float64)

        node_idx = np.ascontiguousarray(np.zeros([x.shape[0]]), dtype=np.int32)

        fea_max_depth = min(self.max_depth, x.shape[1])
        self.split_fea = np.empty(fea_max_depth, dtype=np.int32)
        self.split_thr = np.empty(fea_max_depth, dtype=np.double)

        used_fea = []

        for depth in range(fea_max_depth):
            split_fea, split_thr = find_best_split(x_sliced_sorted, y_sliced_sorted,
                                                   idx_sliced_sorted, node_idx,
                                                   used_fea,
                                                   depth
                                                   )
            # split_fea, split_thr = find_best_split_python(x_sliced_sorted, y_sliced_sorted,
            #                                               idx_sliced_sorted, node_idx,
            #                                               used_fea,
            #                                               depth
            #                                               )

            if split_fea == -1:
                break

            self.split_fea[depth] = self.fea_idx[split_fea]
            self.split_thr[depth] = split_thr

            used_fea.append(split_fea)

            left_obj = np.where(x_sliced_sorted[split_fea] <= split_thr)[0]
            right_obj = np.where(x_sliced_sorted[split_fea] > split_thr)[0]

            node_idx[idx_sliced_sorted[split_fea, left_obj]] *= 2
            node_idx[idx_sliced_sorted[split_fea, left_obj]] += 1

            node_idx[idx_sliced_sorted[split_fea, right_obj]] *= 2
            node_idx[idx_sliced_sorted[split_fea, right_obj]] += 2

        self.node_pred = np.empty(2 ** len(self.split_fea), dtype=np.double)
        self.n_obj_train = np.empty(2 ** len(self.split_fea), dtype=np.int32)
        self._build_tree(x, y, 0, 0)

        return self

    def _build_tree(self, x, y, node_idx, depth, pred=None, n_obj=None):
        node_bias = (1 << depth) - 1
        if pred is not None:
            if depth == len(self.split_fea):
                self.node_pred[node_idx - node_bias] = pred
                self.n_obj_train[node_idx - node_bias] = n_obj
                return
            self._build_tree(None, None, 2 * node_idx + 1, depth + 1, pred, n_obj)
            self._build_tree(None, None, 2 * node_idx + 2, depth + 1, pred, n_obj)
            return

        if depth == len(self.split_fea):
            self.node_pred[node_idx - node_bias] = np.mean(y)
            self.n_obj_train[node_idx - node_bias] = x.shape[0]
            return

        left_obj = np.where(x[:, self.split_fea[depth]] <= self.split_thr[depth])[0]
        right_obj = np.where(x[:, self.split_fea[depth]] > self.split_thr[depth])[0]
        if (
                x.shape[0] < self.min_samples_split or
                left_obj.shape[0] < self.min_samples_leaf or right_obj.shape[0] < self.min_samples_leaf
        ):
            self._build_tree(None, None, 2 * node_idx + 1, depth + 1, np.mean(y), x.shape[0])
            self._build_tree(None, None, 2 * node_idx + 2, depth + 1, np.mean(y), x.shape[0])
        else:
            self._build_tree(x[left_obj], y[left_obj], 2 * node_idx + 1, depth + 1)
            self._build_tree(x[right_obj], y[right_obj], 2 * node_idx + 2, depth + 1)
        return

    def scale_leafs(self, type=None, k=1000.):
        # Regularize tree using leafs scaling
        if type is None:
            leafs_scales = 1.
        elif type == 'sqrt':
            leafs_scales = np.sqrt(self.n_obj_train / (self.n_obj_train + k))
        elif type == 'log':
            leafs_scales = np.log(1. + self.n_obj_train / (self.n_obj_train + k))
        elif type == 'no_k':
            leafs_scales = self.n_obj_train / (1. + self.n_obj_train + np.sqrt(self.n_obj_train))
        self.update_leafs(1., leafs_scales)

    def predict(self, x):
        tree_path = (x[:, self.split_fea] <= self.split_thr)
        return self.node_pred[np.dot(~tree_path, 1 << np.arange(tree_path.shape[-1])[::-1])]

    def update_leafs(self, learning_rate, gamma):
        self.node_pred *= learning_rate * gamma
