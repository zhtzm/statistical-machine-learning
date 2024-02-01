import numpy as np


def sign(X: np.array):
    X_sign = np.ones(X.shape[0], dtype=int)
    for (i, y) in enumerate(X):
        if y <= 0.:
            X_sign[i] = -1
    return X_sign


def sigmoid(X: np.array):
    X_exp_1 = np.exp(X) + 1
    X_sigmoid = np.array([1 / a for a in X_exp_1])
    return X_sigmoid
