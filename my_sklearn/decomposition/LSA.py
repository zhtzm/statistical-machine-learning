import numpy as np

from my_sklearn.utils.NMF import NMF
from my_sklearn.utils.SVD import truncated_SVD


class LSA(object):
    def __init__(self, X, k, method='SVD', t=1000):
        self.X = X.T
        self.k = k
        self.t = t
        if method == 'SVD':
            self.method = self.SVD
        elif method == 'NMF':
            self.method = self.NMF
        else:
            raise ValueError(f'{method} is error')

        self.T = None
        self.Y = None
        self.TFIDF = np.zeros_like(self.X)

    def fit(self):
        self.method()

    def SVD(self):
        U, S, Vt = truncated_SVD(self.X, self.k)
        self.T = U
        self.Y = S @ Vt

    def NMF(self):
        self.T, self.Y = NMF(self.X, self.k, self.t)
