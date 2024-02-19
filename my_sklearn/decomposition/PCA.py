import numpy as np
from numpy import ndarray


class PCA(object):
    def __init__(self, X: ndarray):
        self.X = X
        self.normalized_data = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self):
        mean = np.mean(self.X, axis=0)
        std_dev = np.std(self.X, axis=0)
        self.normalized_data = (self.X - mean) / std_dev

        self.cov_matrix = np.cov(self.normalized_data, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)

    def transform(self, k):
        top_k_eigenvectors = self.eigenvectors[:, :k]
        transformed_data = np.dot(self.normalized_data, top_k_eigenvectors)
        return transformed_data
