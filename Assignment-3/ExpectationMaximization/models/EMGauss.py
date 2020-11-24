from sklearn.cluster import KMeans

import numpy as np
from scipy.stats import multivariate_normal

from models.PCA import covariance_matrix


def multivariate_gaussian(X, mean, cov):  # (N,D) (D,) (D,D)

    mu = mean
    sigma = cov

    if X.shape[1] != mu.shape[0] or (X.shape[1], X.shape[1]) != sigma.shape:
        raise ValueError("The dimensions of the input don't match")
    if np.linalg.det(sigma) == 0:
        raise ValueError("The covariance matrix can't be singular")

    D = len(mu)
    # sigma2 = sigma #np.diag(sigma)
    X = X - mu
    p = (1 / np.sqrt(((2 * np.pi) ** D) * np.linalg.det(sigma))) * np.exp(
        -0.5 * np.sum(X.dot(np.linalg.pinv(sigma)) * X,
                      axis=1))

    return p


class EMGauss:
    def __init__(self, n_components=2, n_iterations=30, eps=1e-9, init=None, early_stop=True):
        self.K = n_components
        self.n_iterations = n_iterations
        self.eps = eps
        self.init = init
        self.early_stop = early_stop

        self.mu = None
        self.sigma = None
        self.w = None

        self.p = None

        self.X = None
        self.likelihood = None

    def fit(self, X, verbose=False, verbose_freq=5):
        self.X = X

        ## initialization

        if self.init == 'kmeans':
            self.mu, self.sigma, self.w = EMGauss.kmeans_mean_cov_init(X, n_clusters=self.K)
        else:
            self.mu = X[np.random.choice(X.shape[0], size=self.K, replace=False),
                      :]  # np.random.randn(self.K, X.shape[1])
            self.sigma = [covariance_matrix(X)] * self.K  # np.random.randn(self.K, X.shape[1], X.shape[1])
            self.w = np.ones((self.K)) / self.K

        self.p = np.ones(shape=(X.shape[0], self.K))  ## N, K

        for j in range(self.n_iterations):
            likelihood = self.likelihood
            self.likelihood = self.__lg_likelihood()

            if verbose and j % verbose_freq == 0:
                print('iteration:', j, 'likelihood:', self.likelihood, 'w:', self.w)  # , 'mu:', self.mu

            if self.early_stop and likelihood is not None and (
                    self.likelihood - likelihood) < self.eps:
                print('early stopping at iteratiion %d with likelihood: %f' % (j, self.likelihood))
                break

            self.__e_step()
            self.__m_step()

        print('Final:', 'likelihood:', self.likelihood, 'w:', self.w)  # , 'mu:', self.mu

    def __e_step(self):
        for k in range(self.K):
            self.p[:, k] = self.w[k] * EMGauss._dist(self.X, mu=self.mu[k], sigma=self.sigma[k])

        self.p = self.p / self.p.sum(axis=1, keepdims=True)

    def __m_step(self):
        # X = N, D
        # p = N, K
        for k in range(self.K):
            self.mu[k] = (self.p[:, [k]] * self.X).sum(axis=0) / self.p[:, k].sum()
            std = self.p[:, [k]] * (self.X - self.mu[k])  # (N,1)*(N,D) -> (N,D)
            self.sigma[k] = (self.X - self.mu[k]).T.dot(std) / self.p[:, k].sum()  # (N,D).T*(N,D) -> (D,D)

            self.w[k] = self.p[:, k].sum() / self.p.shape[0]

    def __lg_likelihood(self):
        probs = np.zeros(self.X.shape[0])
        for k in range(self.K):
            probs += self.w[k] * EMGauss._dist(self.X, mu=self.mu[k], sigma=self.sigma[k])

        return np.log(probs).sum()

    def print_parameters(self):
        for k in range(self.K):
            print('Cluster:', k, 'Ratio:', self.w[k])
            print('mu_%d:' % k, self.mu[k])
            print('sigma_%d:' % k, self.sigma[k])
            print()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        p = np.zeros(shape=(X.shape[0], self.K))  # (N, K)
        for k in range(self.K):
            p[:, k] = self.w[k] * EMGauss._dist(X, mu=self.mu[k], sigma=self.sigma[k])
        return p

    @staticmethod
    def _dist(X, **kwargs):
        # return multivariate_normal.pdf(X, mean=kwargs['mu'], cov=kwargs['sigma'])
        return multivariate_gaussian(X, mean=kwargs['mu'], cov=kwargs['sigma'])

    @staticmethod
    def kmeans_mean_cov_init(X, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=100, algorithm='auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)

        sigma = []
        w = np.zeros(n_clusters)

        for k in range(n_clusters):
            sigma.append(covariance_matrix(X[prediction == k, :]))
            w[k] = (prediction == k).sum()
        w = w / w.sum()
        return kmeans.cluster_centers_, sigma, w
