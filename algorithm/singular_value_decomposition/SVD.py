import numpy as np


def svd(M):
    """
    Args:
        M: numpy matrix of shape (m, n)
    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u, s, v = np.linalg.svd(M)

    return u, s, v
