import numpy as np
from numpy import ndarray

from my_sklearn.model import Model


class NaiveBayes(Model):
    def __init__(self, lmd=0.):
        """
        初始朴素贝叶斯模型
        :param lmd: 平滑量取值
        """
        super().__init__()
        self.features = {}
        self.labels = None
        self.lmd = lmd
        self.pp = {}  # prior_probability
        self.cp = {}  # conditional_probability

    def fit(self, X_train: ndarray, Y_train: ndarray):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        self.labels = list(set(Y_train))
        K = len(self.labels)  # Y的标签值的个数
        # 接下来计算先验概率
        for i in range(K):
            self.pp[self.labels[i]] = \
                (len(Y_train[Y_train == self.labels[i]]) + self.lmd) / (len(Y_train) + self.lmd * K)

        S_n = X_train.shape[1]  # 样本的特征数
        S_j = []  # 用于记录每一个特征的取值的个数
        # 这里统计每个标签的取值及其个数
        for i in range(S_n):
            X_i = X_train[:, i]
            X_i.reshape((-1, 1))
            self.features[i] = list(set(X_i))
            S_j.append(len(self.features[i]))

        # 计算条件概率分布
        for k in range(K):
            a_k = {}  # 用于保存对应label下各个取值的概率
            for j in range(S_n):
                a_k_j = {}  # 用于保存对应label下第j个特征不同取值的概率
                for l in range(S_j[j]):
                    # 计算分母，即训练集中Y中label为第k个label的数量与平滑量的和
                    denominator = len(Y_train[Y_train == self.labels[k]]) + self.lmd * S_j[j]
                    Y_index = np.where(Y_train == self.labels[k])
                    X_j_Y_k = X_train[Y_index, j]
                    select = np.array(X_j_Y_k)
                    # 计算分子，即训练集中X中label为第k个label的数量且第j个特征取值为第l个的数量与平滑量的和
                    numerators = len(select[select == self.features[j][l]]) + self.lmd
                    p_ajl_ck = numerators / denominator
                    a_k_j[self.features[j][l]] = p_ajl_ck
                a_k[j] = a_k_j
            self.cp[self.labels[k]] = a_k

    def predict(self, X: ndarray):
        assert X.shape[1] == len(self.features) and X.ndim == 2

        Y = []
        for x in X:
            label_hat = None
            max_p = 0
            # 用于计算第k个预测label的值
            for k in range(len(self.labels)):
                p = 1 * self.pp[self.labels[k]]  # 先验概率
                # 求先验概率与不同特征特定取值的概率的乘积
                for j in range(len(self.features)):
                    p *= self.cp[self.labels[k]][j][x[j]]
                if max_p < p:
                    max_p = p
                    label_hat = self.labels[k]
            Y.append(label_hat)
        return np.array(Y)
