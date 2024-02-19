import random

import numpy as np
from numpy import ndarray

from my_sklearn.cluster.agglomerative import Agglomerative


class KMeans(object):
    def __init__(self, X: ndarray, k):
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.k = k
        self.center = None
        self.C = None

    def fit(self, method='random'):
        if method == 'random':
            self.random_init_m()
        elif method == 'agglomerative':
            self.agglomerative_init_m()
        else:
            raise ValueError(f"{method} 不是有效的选择")

        pre_cost = None
        cost = float('inf')

        while pre_cost is None or pre_cost == float('inf') or abs(cost - pre_cost) / pre_cost > 0.01:
            pre_cost = cost

            D = np.zeros(shape=(self.n, self.k))
            for i, x in enumerate(self.X):
                deviation = self.center - x
                deviation = np.sum(deviation ** 2, axis=1)
                deviation = deviation.T
                D[i] = deviation

            C = {key: [] for key in range(self.k)}
            for i, x in enumerate(D):
                G_index = np.argmin(x)
                C[G_index].append(i)

            self.C = C
            self.update_m()
            cost = self.cost()

    def cost(self):
        W = 0
        for i in range(self.k):
            G_i = np.array(self.X[self.C[i]])
            ic_lp2_2 = G_i @ np.array(self.center[i]).T
            W += np.sum(ic_lp2_2.T)
        return W

    def update_m(self):
        center = np.zeros(shape=(self.k, self.m))
        for i in range(self.k):
            G_i = np.array(self.X[self.C[i]])
            m_i = np.mean(G_i, axis=0)
            center[i] = m_i
        self.center = center

    def random_init_m(self):
        index = random.sample(range(self.n), self.k)
        # index = list(range(self.k))
        self.center = np.array(self.X[index])

    def agglomerative_init_m(self):
        self.center = np.zeros(shape=(self.k, self.m))

        agglomerative = Agglomerative(self.X)
        agglomerative.fit()
        Gs = agglomerative.k_clusters(self.k)

        for i in range(self.k):
            self.center[i] = self.X[list(Gs[i])[0]]


if __name__ == '__main__':
    test = np.array([[0, 0, 1, 5, 5],
                     [2, 0, 0, 0, 2]])
    model = Kmeans(test.T, 2)
    model.fit()
    print(model.C)
