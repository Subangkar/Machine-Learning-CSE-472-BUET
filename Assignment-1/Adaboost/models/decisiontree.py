# %%
import numpy as np
from datetime import datetime
import collections

from utils import perf_metrics_2X2


class DtUtils:

	@staticmethod
	def resample(X, y, k=None, sample_weight=None):
		if X.shape[0] != y.shape[0]:
			raise Exception('dimension mismatch')
		if k is None:
			k = X.size
		if sample_weight is None:
			sample_weight = np.full((k, 1), 1 / k)
		choices = np.random.choice(X.shape[0], size=k, p=sample_weight)
		return X[choices], y[choices]

	# y = {0,1}
	@staticmethod
	def entropy(y):
		# N = len(y)
		N = y.shape[0]
		s1 = (y == 1).sum()
		if 0 == s1 or N == s1:
			return 0
		p1 = float(s1) / N
		p0 = 1 - p1
		return -p0 * np.log2(p0) - p1 * np.log2(p1)

	# y = {0,1}
	"""
	:returns (information_gain, entropy_left, entropy_right)
	"""
	@staticmethod
	def information_gain(x_parent, y_parent, split, entropy_parent=None):
		y_child_le = y_parent[x_parent <= split]
		y_child_gt = y_parent[x_parent > split]
		N = y_parent.shape[0]
		N_le = y_child_le.shape[0]
		if N_le == 0 or N_le == N:
			return 0
		p_le = float(N_le) / N
		p_gt = 1 - p_le

		entropy_child_le = DtUtils.entropy(y_child_le)
		entropy_child_gt = DtUtils.entropy(y_child_gt)

		if entropy_parent is None:
			entropy_parent = DtUtils.entropy(y_parent)

		return entropy_parent - p_le * entropy_child_le - p_gt * entropy_child_gt

	# info_gain = entropy_parent - p_le * entropy_child_le - p_gt * entropy_child_gt
	# return info_gain, entropy_child_le, entropy_child_gt

	@staticmethod
	def find_split(X, y, column, entropy_parent=None):
		# Binarization -------------------
		sort_idx = np.argsort(X[:, column])
		x_values = X[sort_idx, column]
		y_values = y[sort_idx]
		split_indices = np.nonzero(y_values[1:] != y_values[:-1])[0]
		split_values = np.unique((x_values[split_indices] + x_values[split_indices + 1]) / 2)
		# --------------------------------

		max_ig, split_thresh = max(
			list(map(lambda split_value: (
				DtUtils.information_gain(x_values, y_values, split_value, entropy_parent=entropy_parent), split_value),
			         split_values)))
		return max_ig, split_thresh

	@staticmethod
	def split_dataset(X, y, feature, split_thresh):
		X_left, X_right = X[X[:, feature] <= split_thresh], X[X[:, feature] > split_thresh]
		y_left, y_right = y[X[:, feature] <= split_thresh], y[X[:, feature] > split_thresh]

		return X_left, X_right, y_left, y_right


class TreeNode:
	def __init__(self, depth=1, max_depth=None):
		self.depth = depth
		self.max_depth = max_depth

		self.feature = None
		self.split_threshold = None
		self.prediction = None

		self.left = None
		self.right = None

		self.entropy = None

	def make_terminal_node(self, X, y):
		self.feature = None
		self.split_threshold = None
		self.left = self.right = None
		self.prediction = collections.Counter(y).most_common()[0][0]

	def make_cutoff_node(self, X, y, feature, split_thresh):
		self.feature = feature
		self.split_threshold = split_thresh
		self.left = self.right = None

		self.prediction = (collections.Counter(y[X[:, feature] <= split_thresh]).most_common()[0][0],
		                   collections.Counter(y[X[:, feature] > split_thresh]).most_common()[0][0])

	def buildtree(self, X, y, depth, max_depth=None):
		# print(depth, end=' ')
		self.entropy = DtUtils.entropy(y)
		if len(y) == 1 or self.entropy == 0:
			self.make_terminal_node(X, y)
			return

		best_feature, max_ig, best_split_thresh = max(
			list(map(lambda feature: ((feature,) + (DtUtils.find_split(X, y, feature, entropy_parent=self.entropy))),
			         range(X.shape[1]))),
			key=lambda v: v[1])

		if max_ig <= 0:
			self.make_terminal_node(X, y)
			return

		if max_depth is not None and depth >= max_depth:
			self.make_cutoff_node(X, y, feature=best_feature, split_thresh=best_split_thresh)
			return

		self.feature = best_feature
		self.split_threshold = best_split_thresh

		X_left, X_right, y_left, y_right = DtUtils.split_dataset(X, y, best_feature, best_split_thresh)

		self.left = TreeNode(self.depth + 1, self.max_depth)
		self.right = TreeNode(self.depth + 1, self.max_depth)

		self.left.buildtree(X=X_left, y=y_left, depth=depth + 1, max_depth=max_depth)
		self.right.buildtree(X=X_right, y=y_right, depth=depth + 1, max_depth=max_depth)

	# loosely ok
	def predict_val(self, x):
		if self.feature is None:
			return self.prediction
		if x[self.feature] <= self.split_threshold:
			return self.left.predict_val(x) if self.left is not None else self.prediction[0]
		else:
			return self.right.predict_val(x) if self.right is not None else self.prediction[1]

	# using loop
	def predict(self, X):
		return np.array(list(map(lambda x: self.predict_val(x), X)))

	@staticmethod
	def build_dtree(X, y, depth, max_depth=None):
		node = TreeNode()
		node.entropy = DtUtils.entropy(y)

		if len(y) == 1 or (max_depth is not None and depth >= max_depth) or node.entropy == 0:
			return node.make_terminal_node(X, y)

		best_feature, max_ig, best_split_thresh = max(
			list(map(lambda feature: ((feature,) + (DtUtils.find_split(X, y, feature))), range(X.shape[1]))),
			key=lambda v: v[1])

		if max_ig == 0:
			return node.make_terminal_node(X, y)

		node.feature = best_feature
		node.split = best_split_thresh

		X_left, X_right, y_left, y_right = DtUtils.split_dataset(X, y, best_feature, best_split_thresh)

		node.left = TreeNode.build_dtree(X=X_left, y=y_left, depth=depth + 1, max_depth=max_depth)
		node.right = TreeNode.build_dtree(X=X_right, y=y_right, depth=depth + 1, max_depth=max_depth)

		return node

	@staticmethod
	def print_tree(node, depth):
		if node.feature is not None:
			TreeNode.print_tree(node.left, depth + 1)
			print('     ' * depth + 'X[' + str(node.feature) + ']' + '<=' + '{:.3f}'.format(
				node.split_threshold) + ' ent=' + '{:.3f}'.format(node.entropy))
			TreeNode.print_tree(node.right, depth + 1)
		else:
			print('     ' * depth + 'y=' + str(int(node.prediction)) + ' ent=' + '{:.3f}'.format(node.entropy))


class DecisionTree:
	def __init__(self, max_depth=None):
		self.max_depth = max_depth

		self.root = None

	def fit(self, X, y, sample_weight=None):
		X = np.array(X)
		y = np.array(y)
		if sample_weight is not None:
			N = sample_weight.shape[0]
			X, y = DtUtils.resample(X, y, N, sample_weight)
		self.root = TreeNode(max_depth=self.max_depth)
		self.root.buildtree(X=X, y=y, depth=0, max_depth=self.max_depth)

	def predict(self, X):
		return self.root.predict(np.array(X))

	def score(self, X, y):
		y_p = self.predict(X)
		return np.mean(y_p == y)

	def report(self, X, y):
		# return classification_report(np.array(y), self.predict(X), target_names=['class 0', 'class 1'])
		return perf_metrics_2X2(y_true=np.array(y), y_pred=self.predict(X))

	def print_tree(self):
		TreeNode.print_tree(self.root, depth=0)


# %%
def get_data():
	dataset = np.array([[1, 0, 1],
	                    [1, 1, 1],
	                    [0, 0, 1],
	                    [0, 1, 0]])
	return dataset[:, :-1], dataset[:, -1]


if __name__ == '__main__':
	X, Y = get_data()

	# try donut and xor
	# from sklearn.utils import shuffle
	# X, Y = get_xor()
	# # X, Y = get_donut()
	# X, Y = shuffle(X, Y)

	# only take 0s and 1s since we're doing binary classification
	idx = np.logical_or(Y == 0, Y == 1)
	X = X[idx]
	Y = Y[idx]

	# split the data
	Ntrain = len(Y) // 2
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	model = DecisionTree()
	# model = DecisionTree(max_depth=7)
	t0 = datetime.now()
	model.fit(Xtrain, Ytrain)
	print("Training time:", (datetime.now() - t0))

	t0 = datetime.now()
	print("Train accuracy:", model.score(Xtrain, Ytrain))
	print("Time to compute train accuracy:", (datetime.now() - t0))

	t0 = datetime.now()
	print("Test accuracy:", model.score(Xtest, Ytest))
	print("Time to compute test accuracy:", (datetime.now() - t0))
