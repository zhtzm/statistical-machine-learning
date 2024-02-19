from numpy import ndarray
from copy import deepcopy


class AgglomerativeClustering(object):
    def __init__(self, X: ndarray):
        super().__init__()
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.D = None
        self.layers = []

    def fit(self):
        self.D = self.X @ self.X.T
        K = self.n

        Gs = [set(i) for i in range(K)]
        self.layers.append(deepcopy(Gs))

        while K > 1:
            Gi_1 = 0
            Gi_2 = 0
            G_min_distance = float('inf')

            for i in range(K):
                for j in range(i + 1, K):
                    ij_distance = self.cluster_distance(Gs[i], Gs[j])
                    if ij_distance < G_min_distance:
                        Gi_1 = i
                        Gi_2 = j
                        G_min_distance = ij_distance

            G1 = Gs.pop(Gi_1)
            G2 = Gs.pop(Gi_2)
            G3 = G1.union(G2)
            Gs.append(G3)

            self.layers.append(deepcopy(Gs))
            K -= 1

    def k_clusters(self, k):
        assert k > 1
        return self.layers[self.n - k]

    def cluster_distance(self, G1: set, G2: set):
        min_distance = float('inf')
        for i in G1:
            for j in G2:
                if self.D[i][j] < min_distance:
                    min_distance = self.D[i][j]

        return min_distance
