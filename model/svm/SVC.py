import numpy as np
from numpy import ndarray

from model.model import Model
from utils.kernel import gaussian_kernel


class SVC(Model):
    def __init__(self, C, sigma, max_iter):
        super().__init__()
        self.alpha = None
        self.b = 0.
        self.X_train = None
        self.Y_train = None
        self.N = 0
        self.K = None
        self.eCache = None

        self.max_iters = max_iter
        self.C = C
        self.sigma = sigma

    def fit(self, X_train: ndarray, Y_train: ndarray):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        self.X_train = X_train
        self.Y_train = Y_train
        self.N = X_train.shape[0]
        self.alpha = np.zeros(X_train.shape[0])
        self.K = self._kernel_matrix()
        self.eCache = np.zeros((self.N, 2))

        self.SMO()

    def predict(self, X: ndarray):
        assert X.shape[1] == self.X_train.shape[1] and X.ndim == 2

        Y_hat = []
        for x in X:
            y_hat = self.g(x)
            Y_hat.append(y_hat)
        return np.array(Y_hat)

    def SMO(self):
        iters = 0
        complete = True  # 标志是否应该遍历整个数据集
        alphaPairsChanged = 0  # 标志一次循环中α更新的次数

        while iters < self.max_iters and ((alphaPairsChanged > 0) or complete):
            alphaPairsChanged = 0
            if complete:
                for i in range(self.N):
                    alphaPairsChanged += self.inner(i)  # 调用内循环
                    print("full dataset, iter: %d i:%d,pairs changed:%d" % (iters, i, alphaPairsChanged))
                iters += 1
            else:
                nonBounds = np.where((self.alpha > 0) & (self.alpha < self.C))[0]  # 获取非边界值中的索引
                for i in nonBounds:
                    alphaPairsChanged += self.inner(i)
                    print("non bound, iter: %d i:%d,pairs changed:%d" % (iters, i, alphaPairsChanged))
                iters += 1

            if complete:
                complete = False
            elif alphaPairsChanged == 0:
                complete = True

            print("iteration number: %d" % iters)

    def inner(self, i):
        Ei = self.E(i)

        y_g = self.Y_train[i] * self.g(self.X_train[i])
        if (y_g < 1 and self.Y_train[i] < self.C) or (y_g > 1 and self.Y_train[i] > self.C):
            j, Ej = self.select_j(i, Ei)
            old_alpha_i = self.alpha[i]  # deepcopy(self.alpha[i])
            old_alpha_j = self.alpha[j]  # deepcopy(self.alpha[j])

            if self.Y_train[i] != self.Y_train[j]:
                L = max(0, old_alpha_j - old_alpha_i)
                H = min(self.C, self.C + old_alpha_j - old_alpha_i)
            else:
                L = max(0, old_alpha_j + old_alpha_i - self.C)
                H = min(self.C, old_alpha_j + old_alpha_i)
            if L == H:
                return 0

            eta = self.K[i][i] + self.K[j][j] - 2 * self.K[i][j]
            if eta <= 0:
                print("eta <= 0")
                return 0

            new_unc = old_alpha_j + self.Y_train[j] * (Ei - Ej) / eta
            if new_unc > H:
                self.alpha[j] = H
            elif new_unc < L:
                self.alpha[j] = L
            else:
                self.alpha[j] = new_unc
            self.update_Ek(j)

            if abs(self.alpha[j] - old_alpha_j) < 0.00001:
                return 0  # 因为α_2变化量较小，所以我们没有必要非得把值变回原来的旧值
            self.alpha[i] = old_alpha_i + self.Y_train[i] * self.Y_train[j] * (old_alpha_j - self.alpha[j])
            self.update_Ek(i)

            bi = -Ei - self.Y_train[i] * self.K[i][i] * (self.alpha[i] - old_alpha_i) \
                 - self.Y_train[j] * self.K[j][i] * (self.alpha[j] - old_alpha_j) + self.b
            bj = -Ej - self.Y_train[i] * self.K[i][j] * (self.alpha[i] - old_alpha_i) \
                 - self.Y_train[j] * self.K[j][j] * (self.alpha[j] - old_alpha_j) + self.b

            if 0 < self.Y_train[i] < self.C:
                self.b = bi
            elif 0 < self.Y_train[j] < self.C:
                self.b = bj
            else:
                self.b = 1 / 2 * (bi + bj)

            return 1

        else:
            return 0

    def select_j(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = np.array([1, Ei])
        validEcacheList = np.where(self.eCache[:, 0] == 1)
        if (len(validEcacheList[0])) > 1:
            for k in validEcacheList[0]:
                if k == i:
                    continue
                Ek = self.E(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = self._select_j_rand(i)
            Ej = self.E(j)
        return j, Ej

    def update_Ek(self, k):
        Ek = self.E(k)
        self.eCache[k] = np.array([1, Ek])

    def E(self, i):
        return self.g(self.X_train[i]) - self.Y_train[i]

    def g(self, x):
        y_hat = self.b
        for i in range(self.N):
            y_hat += self.alpha[i] * self.Y_train[i] * gaussian_kernel(x, self.X_train[i], self.sigma)
        return y_hat

    def _select_j_rand(self, i):
        j = i
        while j == i:
            j = int(np.random.uniform(0, self.N))
        return j

    def _kernel_matrix(self):
        K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                K[i][j] = gaussian_kernel(self.X_train[i], self.X_train[j], self.sigma)
        return K
