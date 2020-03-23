import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import copy

from models.decisiontree import DecisionTree
from utils import perf_metrics_2X2, plot_confusion_matrix


class AdaBoost:
	def __init__(self, n_estimators=None, base_estimator=None):
		if n_estimators is None:
			n_estimators = 5
		if base_estimator is None:
			base_estimator = DecisionTree(max_depth=1)

		self.n_estimators = n_estimators
		self.classifier = base_estimator

		self.n_classes = 0
		self.classes = None
		self.class_binary_encoded = None

		self.estimators = []
		self.estimator_weights = []

	def fit(self, X, y, eps=1E-12, random_state=None):
		"""
		Only for binary classifier
		:param random_state: seed for random sampling
		:param X: feature vectors
		:param y: target vector
		:param eps: precision to avoid division by 0
		"""
		N, _ = X.shape
		W = np.ones(N) / N

		self.classes = np.unique(y)
		self.n_classes = self.classes.size
		self.class_binary_encoded = OneHotEncoder(categories=self.classes, sparse=False)

		if random_state is not None:
			np.random.seed(random_state)

		print('fitting ' + str(self.n_estimators) + ' models')
		for k in range(self.n_estimators):
			estimator = copy.copy(self.classifier)
			estimator.fit(X, y, sample_weight=W, random_state=None)
			y_p = estimator.predict(X)

			error = W.dot(y_p != y)  # error = sum(w[j]) if y_p_j != y_j

			if error > 0.5:
				continue

			if error == 0:
				error = eps

			# PDF
			W[y_p == y] *= (error / (1 - error))
			W /= W.sum()

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
		N, _ = X.shape
		y_val = np.zeros(N)
		for estimator_weight, estimator in zip(self.estimator_weights, self.estimators):
			y_p = estimator.predict(X)
			y_p[y_p == 0] = -1
			y_val += estimator_weight * y_p
		return (np.sign(y_val) == 1).astype(int)  # , y_val

	# class_probs = np.zeros(shape=(N, self.n_classes))
	# for estimator_weight, estimator in zip(self.estimator_weights, self.estimators):
	# 	y_p = estimator.predict(X)
	# 	# y_p_reshaped = y_p.reshape(-1, 1)
	# 	feature_select_vect = self.class_binary_encoded.fit_transform(y_p.reshape(-1, 1))
	# 	class_probs += estimator_weight * feature_select_vect

	def score(self, X, y):
		return np.mean(self.predict(X) == y)

	def report(self, X, y):
		# return classification_report(np.array(y), self.predict(X), target_names=['class 0', 'class 1'])
		return perf_metrics_2X2(y_true=np.array(y), y_pred=self.predict(X))

	def plot_cm(self, X, y):
		plot_confusion_matrix(y_true=y, y_pred=self.predict(X))


if __name__ == '__main__':

	def get_data():
		dataset = np.array([[1, 0, 1],
		                    [1, 1, 1],
		                    [0, 0, 1],
		                    [0, 1, 0]])
		return dataset[:, :-1], dataset[:, -1]


	X, Y = get_data()
	# Y[Y == 0] = -1  # make the targets -1,+1
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
