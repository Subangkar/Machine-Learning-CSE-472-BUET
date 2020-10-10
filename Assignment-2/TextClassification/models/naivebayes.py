import numpy as np


class NaiveBayes:
    def __init__(self, smoothing_factor=1.00):
        self.vocab_size = 0
        self.smoothing_factor = smoothing_factor
        self.classes = None
        self.class_priors = None
        self.class_word_count = None
        self.class_word_prob = None

    def fit(self, X: np.array, y):
        self.vocab_size = X.shape[1]
        self.classes = np.unique(y)
        # self.class_priors = {k: (y == k).sum() / len(y) for k in self.classes}
        self.class_priors = np.array([(y == k).mean() for k in self.classes])
        self.class_word_count = np.zeros((len(self.classes), self.vocab_size))  # K,V
        for k in self.classes:
            indices = np.where(y == k)[0]
            self.class_word_count[k, :] = X[indices, :].sum(axis=0)

        self.class_word_prob = (self.class_word_count + self.smoothing_factor) / (
                self.class_word_count.sum(axis=0, keepdims=True) + self.smoothing_factor * self.vocab_size)

    def predict(self, X, return_probs=False):
        y_pred = np.zeros(X.shape[0], dtype=int)
        y_prob = np.zeros((X.shape[0], len(self.classes))) if return_probs else None

        for i, x in enumerate(X):
            words_indices = np.where(x > 0)[0]
            p_ks_given_X = self.class_priors * self.class_word_prob[:, words_indices].prod(axis=1)
            y_pred[i] = np.argmax(p_ks_given_X)
            if return_probs:
                y_prob[i] = p_ks_given_X / p_ks_given_X.sum()

        return (y_pred, y_prob) if return_probs else y_pred

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return (y == y_pred).mean()
