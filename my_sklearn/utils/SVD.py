import numpy as np


def truncated_SVD(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    if k < S.shape[0]:
        U_truncated = U[:, :k]
        S_truncated = np.diag(S[:k])
        Vt_truncated = Vt[:k, :]
    else:
        U_truncated = U
        S_truncated = S
        Vt_truncated = Vt

    return U_truncated, S_truncated, Vt_truncated


def SVD(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    return U, S, Vt
