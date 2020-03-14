from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import copy


# from rf_classification import get_data


class AdaBoost:
	def __init__(self, n_estimators=None, base_estimator=None):
		if n_estimators is None:
			n_estimators = 5
		if base_estimator is None:
			base_estimator = DecisionTreeClassifier(max_depth=1)

		self.n_estimators = n_estimators
		self.classifier = base_estimator

		self.estimators = []
		self.estimator_weights = []

	def fit(self, X, y, eps=1E-10):
		N, _ = X.shape
		W = np.ones(N) / N

		print('fitting ' + str(self.n_estimators) + ' models')
		for k in range(self.n_estimators):
			estimator = copy.copy(self.classifier)
			estimator.fit(X, y, sample_weight=W)
			y_p = estimator.predict(X)

			error = W.dot(y_p != y)  # error = sum(w[j]) if y_p_j != y_j

			if error > 0.5:
				continue

			if error == 0:
				error = eps

			# PDF
			W[y_p == y] *= (error / (1 - error))
			W /= (W.sum() + eps)

			estimator_weight = np.log(1 - error) - np.log(error)
			# ------------------------------------------

			# https://towardsdatascience.com/boosting-algorithm-adaboost-b6737a9ee60c
			# estimator_weight = 0.5 * (np.log(1 - error) - np.log(error))
			#
			# W = W * np.exp(-estimator_weight * y * y_p)
			# W = W / W.sum()
			# -----------------------------------------------------------------------

			self.estimators.append(estimator)
			self.estimator_weights.append(estimator_weight)

	def predict(self, X):
		# NOT like SKLearn API
		# we want accuracy and exponential loss for plotting purposes
		N, _ = X.shape
		FX = np.zeros(N)
		for alpha, tree in zip(self.estimator_weights, self.estimators):
			FX += alpha * tree.predict(X)
		return np.sign(FX), FX

	def score(self, X, y):
		# NOT like SKLearn API
		# we want accuracy and exponential loss for plotting purposes
		y_p, FX = self.predict(X)
		L = np.exp(-y * FX).mean()
		return np.mean(y_p == y), L


def get_data():
	dataset = np.array([[1, 0, 1],
	                    [1, 1, 1],
	                    [0, 0, 1],
	                    [0, 1, 0]])
	return dataset[:, :-1], dataset[:, -1]


if __name__ == '__main__':

	X, Y = get_data()
	Y[Y == 0] = -1  # make the targets -1,+1
	Ntrain = int(0.8 * len(X))
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	T = 100
	train_errors = np.empty(T)
	test_losses = np.empty(T)
	test_errors = np.empty(T)
	for num_trees in range(T):
		if num_trees == 0:
			train_errors[num_trees] = None
			test_errors[num_trees] = None
			test_losses[num_trees] = None
			continue
		if num_trees % 20 == 0:
			print(num_trees)

		model = AdaBoost(num_trees)
		model.fit(Xtrain, Ytrain)
		acc, loss = model.score(Xtest, Ytest)
		acc_train, _ = model.score(Xtrain, Ytrain)
		train_errors[num_trees] = 1 - acc_train
		test_errors[num_trees] = 1 - acc
		test_losses[num_trees] = loss

		if num_trees == T - 1:
			print("final train error:", 1 - acc_train)
			print("final test error:", 1 - acc)

	plt.plot(test_errors, label='test errors')
	plt.plot(test_losses, label='test losses')
	plt.legend()
	plt.show()

	plt.plot(train_errors, label='train errors')
	plt.plot(test_errors, label='test errors')
	plt.legend()
	plt.show()
