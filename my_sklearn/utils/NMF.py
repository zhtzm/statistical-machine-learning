import numpy as np


def NMF(A, k, t):
    # 初始化随机非负矩阵 W 和 H
    m, n = A.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)

    # 迭代更新
    for _ in range(t):
        # 更新矩阵 H
        numerator = np.dot(W.T, A)
        denominator = np.dot(np.dot(W.T, W), H) + 1e-9  # 添加一个小的正数以避免除零错误
        H *= numerator / denominator

        # 更新矩阵 W
        numerator = np.dot(A, H.T)
        denominator = np.dot(W, np.dot(H, H.T)) + 1e-9
        W *= numerator / denominator

    return W, H
