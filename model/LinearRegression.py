import numpy as np
from numpy import ndarray


class LinearRegression(object):
    def __init__(self, reg=None, lmd=None):
        assert reg is None or reg == 'Ridge'
        if reg is not None:
            assert lmd is not None

        self._theta = None

        self.reg = reg
        self.lmd = lmd

    def fit(self, X: ndarray, Y: ndarray):
        assert X.ndim == 2 and Y.ndim == 1
        assert X.shape[0] == Y.shape[0]
        self._theta = self.LSE(X, Y)

    def LSE(self, X: ndarray, Y: ndarray):
        X_hat = np.c_[X, np.ones(X.shape[0])]
        if self.reg is None:
            return np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y.T
        elif self.reg == 'Ridge':
            identity_matrix = np.eye(X.shape[1])
            return np.linalg.inv(X_hat.T @ X_hat + self.lmd * identity_matrix) @ X_hat.T @ Y.T

    def predict(self, X: ndarray):
        assert X.shape[1] == self._theta.shape[0] - 1 and X.ndim == 2

        X_hat = np.c_[X, np.ones(X.shape[0])]
        Y_predict = (X_hat @ self._theta).T

        return Y_predict
