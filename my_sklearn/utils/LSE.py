import numpy as np
from numpy import ndarray


def LSE(X: ndarray, Y: ndarray):
    X_hat = np.c_[X, np.ones(X.shape[0])]
    return np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y.T


def ridge_LSE(X: ndarray, Y: ndarray, lmd):
    X_hat = np.c_[X, np.ones(X.shape[0])]
    identity_matrix = np.eye(X_hat.shape[1])
    return np.linalg.inv(X_hat.T @ X_hat + lmd * identity_matrix) @ X_hat.T @ Y.T
