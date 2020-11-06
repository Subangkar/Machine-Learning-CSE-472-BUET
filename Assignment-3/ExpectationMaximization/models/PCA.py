import numpy as np
from sklearn.preprocessing import StandardScaler


def covariance_matrix(X, mean=None):
    n = X.shape[0]
    mean_vec = np.mean(X, axis=0) if mean is None else mean
    S = (X - mean_vec).T.dot((X - mean_vec)) / (n - 1)
    return S


def pca_2(X, n_components=2):
    X_std = StandardScaler().fit_transform(X)

    cov_mat = covariance_matrix(X_std)

    eigvalues, eigvectors = np.linalg.eig(cov_mat)
    topkeig = eigvectors[:, np.argsort(np.abs(eigvalues))[-n_components:]]  #

    X_t = X.dot(topkeig)
    return X_t
