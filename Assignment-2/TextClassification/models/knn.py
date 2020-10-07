from scipy.stats import mode
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class Knn:
    def __init__(self, n_neighbors, metric='euclidean'):
        self.X_train = None
        self.y_train = None
        self.K = n_neighbors
        self.metric = Knn.dist_func(metric)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        dist = pairwise_distances(X, self.X_train, metric=self.metric)
        indices = np.argsort(dist, axis=1)[:, :self.K]
        y_top = self.y_train[indices]
        y_pred = mode(y_top, axis=1)[0].flatten()
        return y_pred

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return (y == y_pred).mean()

    @staticmethod
    def dist_func(metric):
        _dist_function = {
            'cosine': lambda v1, v2: -np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
            'hamming': lambda v1, v2: np.count_nonzero(v1 != v2),
            'euclidean': lambda v1, v2: np.linalg.norm(v1 - v2),
        }
        return _dist_function[metric]
