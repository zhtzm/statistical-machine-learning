from numpy import ndarray

from model.model import Model
from utils.LSE import ridge_LSE


class Ridge(Model):
    def __init__(self, lmd):
        super().__init__()
        self.w = None
        self.b = 0.
        self.lmd = lmd

    def fit(self, X: ndarray, Y: ndarray):
        assert X.ndim == 2 and Y.ndim == 1
        assert X.shape[0] == Y.shape[0]

        theta = ridge_LSE(X, Y, self.lmd)
        self.w = theta[: -1]
        self.b = theta[-1]

    def predict(self, X: ndarray):
        assert X.shape[1] == self.w.shape[0] and X.ndim == 2

        Y_predict = (X @ self.w + self.b).T
        return Y_predict
