import numpy as np
from numpy import ndarray

from my_sklearn.model import Model
from my_sklearn.utils.data_structure import KDNode
from my_sklearn.utils.data_structure import MaxHeap
from my_sklearn.utils import find_most_label


class KNN(Model):
    def __init__(self, k, p=2, is_kd=False):
        super().__init__()
        self.X_train = None
        self.Y_train = None
        self.kd_tree = None
        self.ndim = None
        self.is_kd = is_kd
        self.k = k
        self.p = p

    def fit(self, X_train: ndarray, Y_train: ndarray):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        self.X_train = X_train
        self.Y_train = Y_train
        self.ndim = self.X_train.shape[1]
        if self.is_kd:
            data = np.concatenate([X_train, Y_train.reshape(-1, 1)], axis=1)
            self.kd_tree = self._build_tree(data=data, axis=0)

    def predict(self, X: ndarray):
        assert X.shape[1] == self.X_train.shape[1] and X.ndim == 2

        Y = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            k_labels = self._first_k_labels(x)
            Y[i] = find_most_label(k_labels)

        return Y

    def _build_tree(self, data, axis, parent=None):
        if len(data) == 0:
            return None
        elif len(data) == 1:
            return KDNode(parent=parent, value=data[0], axis=axis)

        data = np.array(sorted(data, key=lambda x: x[axis]))
        median = len(data) // 2
        median_value = data[median]
        root = KDNode(parent=parent, value=median_value, axis=axis)

        new_axis = (axis + 1) % self.ndim
        left_data = data[:median, :]
        right_data = data[median + 1:, :]

        root.left = self._build_tree(data=left_data, axis=new_axis, parent=root)
        root.right = self._build_tree(data=right_data, axis=axis, parent=root)

        return root

    def _search_tree(self, x: ndarray):
        assert 0 <= self.k <= self.X_train.shape[0]

        heap = MaxHeap()

        def recursion(root):
            if root is None:
                return
            if x[root.axis] < root.data[root.axis]:
                brother = root.right
                if root.left is not None:
                    recursion(root.left)
            else:
                brother = root.left
                if root.right is not None:
                    recursion(root.right)

            distance = np.sum(np.abs((root.data - x) ** self.p)) ** (1 / self.p)
            if len(heap) < self.k:
                heap.push((root.data, root.label, distance))
            else:
                max_distance = heap.peak()[-1]
                if distance < max_distance:
                    heap.pop()
                    heap.push((root.data, root.label, distance))

            if brother is not None:
                max_distance = heap.peak()[-1]
                if len(heap) < self.k or abs(x[root.axis] - root.data[root.axis]) < max_distance:
                    recursion(brother)

        recursion(self.kd_tree)
        labels = [e[1] for e in heap.array[1:]]
        return labels

    def _search_linear(self, x: ndarray):
        result = np.sum(np.abs((self.X_train - x) ** self.p), axis=1) ** (1 / self.p)
        index = np.argpartition(result, kth=self.k)
        labels = self.Y_train[index]
        return labels[: self.k]

    def _first_k_labels(self, x):
        if not self.is_kd:
            labels = self._search_linear(x)
        else:
            labels = self._search_tree(x)

        return labels
