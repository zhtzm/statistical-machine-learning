from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """
    所有模型的超类，定义有训练用的
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X_train, Y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass
