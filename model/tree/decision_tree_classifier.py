import copy

import numpy as np

from model.model import Model
from utils.find_most_label import find_most_label
from utils.Gini import Gini


class DecisionTreeClassifier(Model):
    def __init__(self):
        super().__init__()
        self.n_feature = 0
        self.tree = None
        self.feature_property = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, feature_property=None):
        assert X_train.ndim == 2 and Y_train.ndim == 1
        assert X_train.shape[0] == Y_train.shape[0]

        self.n_feature = X_train.shape[1]

        dataset = np.concatenate([X_train, Y_train.reshape(-1, 1)], axis=1)
        features = list(range(X_train.shape[1]))
        self.feature_property = feature_property
        self.tree = build_tree(dataset, features, feature_property)

    def predict(self, X):
        assert X.shape[1] == self.n_feature and X.ndim == 2

        Y = []
        for x in X:
            root = self.tree
            while 'son' in root.keys():
                feature = root['feature']
                if self.feature_property[feature]:
                    if x[feature] <= root['value']:
                        root = root['son'][root['value']]
                    else:
                        root = root['son']['-others-']
                else:
                    if x[feature] in root['son'].keys():
                        root = root['son'][x[feature]]
                    else:
                        root = root['son']['-others-']
            Y.append(root['predict'])

        return np.array(Y)


def build_tree(dataset, features, feature_property):
    labels = dataset[:, -1]

    # 第一种结束情况
    if len(labels[labels == labels[0]]) == len(labels):
        return {'predict': labels[0]}
    # 第二种结束情况
    if len(features) == 0:
        return {'predict': find_most_label(labels)}

    best_feature, split_value = choose_best_feature(dataset, features, feature_property)
    decision_tree = {'feature': best_feature, 'son': {}}
    sub_features = copy.deepcopy(features)
    sub_features.remove(best_feature)

    if feature_property[best_feature]:
        value_index = np.where(dataset[:, best_feature] <= split_value)
        decision_tree['value'] = split_value
    else:
        value_index = np.where(dataset[:, best_feature] == split_value)
    sub_dataset1 = dataset[value_index[0], :]
    sub_dataset2 = dataset[list(set(range(len(dataset))) - set(value_index[0])), :]

    decision_tree['son'][split_value] = build_tree(sub_dataset1, sub_features, feature_property)
    decision_tree['son']['-others-'] = build_tree(sub_dataset2, sub_features, feature_property)
    return decision_tree


def pruning(tree):
    pass


def choose_best_feature(dataset, features, feature_property):
    bestGini = 1
    best_feature = None
    split_value = None

    for i in features:
        feature_data = dataset[:, i]
        feature_values = set(feature_data)
        flag = feature_property[i]

        for value in feature_values:
            if flag:
                value_index = np.where(feature_data <= value)
            else:
                value_index = np.where(feature_data == value)
            sub_dataset1 = dataset[value_index[0], :]
            sub_dataset2 = dataset[list(set(range(len(feature_data))) - set(value_index[0])), :]

            prob1 = len(sub_dataset1) / float(len(dataset))
            prob2 = len(sub_dataset2) / float(len(dataset))
            gini1 = Gini(sub_dataset1)
            gini2 = Gini(sub_dataset2)
            gini = prob1 * gini1 + prob2 * gini2

            if gini < bestGini:
                bestGini = gini
                best_feature = i
                split_value = value

    return best_feature, split_value
