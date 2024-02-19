from numpy import ndarray

from my_sklearn.model import Model
from my_sklearn.utils import LSE


class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.w = None
        self.b = 0.

    def fit(self, X_train: ndarray, Y_train: ndarray):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        theta = LSE(X_train, Y_train)
        self.w = theta[: -1]
        self.b = theta[-1]

    def predict(self, X: ndarray):
        assert X.shape[1] == self.w.shape[0] and X.ndim == 2

        Y_predict = (X @ self.w + self.b).T
        return Y_predict
