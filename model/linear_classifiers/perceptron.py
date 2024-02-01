import numpy as np

from model.model import Model
from utils.signal_function import sign


class Perceptron(Model):
    def __init__(self, lr: float):
        """
        初始化感知机模型
        :param lr: 学习率
        """
        super().__init__()
        self.w = None  # 权值
        self.b = 0.  # 偏移量
        self.lr = lr  # 学习率

    def fit(self, X_train: np.array, Y_train: np.array, antithesis=False):
        """
        :param X_train: 要求numpy的array数据类型，应为二维
        :param Y_train: 要求numpy的array数据类型，应为一维
        :param antithesis: 是否使用对偶形式训练
        """
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if not antithesis:
            self._train(X_train, Y_train)
        else:
            self._antithesis_train(X_train, Y_train)

    def predict(self, X: np.array) -> np.array:
        """
        :param X: 要求numpy的array数据类型，应为二维
        :return: 返回一维numpy的array作为预测值
        """
        assert X.shape[1] == self.w.shape[0] and X.ndim == 2

        Y_hat = self.w @ X.T + self.b
        Y = sign(Y_hat)
        return Y

    def _train(self, X, Y):
        self.w = np.ones(X.shape[0])

        Y_hat = Y * (self.w @ X.T + self.b)
        index = self._has_error(Y_hat)
        while index is not None:
            self.w += self.lr * Y[index] * X[index]
            self.b += self.lr * Y[index]
            Y_hat = Y * (self.w @ X.T + self.b)
            index = self._has_error(Y_hat)

    def _antithesis_train(self, X, Y):
        gram_matrix = X @ X.T  # gram矩阵，公式化简后所含的的一项
        self.a = np.zeros(X.shape[0])  # 每个数据点作为随机下降的次数

        Y_hat = Y * ((self.a * Y) @ gram_matrix + self.b)
        index = self._has_error(Y_hat)
        while index is not None:
            self.a[index] += self.lr * 1
            self.b += self.lr * Y[index]
            Y_hat = Y * ((self.a * Y) @ gram_matrix + self.b)
            index = self._has_error(Y_hat)

        self.w = (self.a * Y) @ gram_matrix

    @staticmethod
    def _has_error(Y_hat):
        """
        用于返回第一个分类错误数据点下标值
        """
        for i, y_hat in enumerate(Y_hat):
            if y_hat <= 0:
                return i
        return None
