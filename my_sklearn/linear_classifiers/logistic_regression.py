import numpy as np

from my_sklearn.model import Model


class LogisticRegression(Model):
    def __init__(self, lr, r, e=0):
        super().__init__()
        self.w = None
        self.b = 0.
        self.lr = lr
        self.r = r
        self.e = e
        self.X_train = None
        self.Y_train = None
        self.X_ = None
        self.w_ = None

    def fit(self, X_train, Y_train):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_ = np.c_[X_train, np.ones(X_train.shape[0])]
        self.w_ = np.ones(self.X_.shape[1])

        for i in range(self.r):
            self.w_ -= self.lr * self.cal_dw()
            cost = self.cost_function()
            if cost < self.e:
                break

        self.w = self.w_[:, -1]
        self.b = self.w_[-1]

    def predict(self, X):
        assert X.shape[1] == self.w.shape[0] and X.ndim == 2

        Y = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            p = self.sigmoid(self.w @ x.T + self.b)
            q = 1 - p
            if p > q:
                Y[i] = 1

        return Y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def cost_function(self):  # 损失函数
        lost = np.sum(
            self.Y_train * np.log(self.sigmoid(self.w_ @ self.X_.T)) + (1 - self.Y_train) * np.log(
                1 - self.sigmoid(self.w_ @ self.X_.T))) / self.X_.shape[0]
        return -lost

    def cal_dw(self):  # 求w的偏导数
        dw = (self.sigmoid(self.w_ @ self.X_.T) - self.Y_train) @ self.X_train / self.X_train.shape[0]
        return dw
