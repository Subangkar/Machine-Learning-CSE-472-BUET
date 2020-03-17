# %%
import random
import sys
import time
import numpy as np
import pandas as pd

import collections

from scipy.stats import hmean
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import copy

# %%
random.seed(1)


# sys.stdout = open('report.log', 'w')

# %%
def perf_metrics_2X2(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	TN = cm[0][0]
	FN = cm[1][0]
	TP = cm[1][1]
	FP = cm[0][1]

	precision = TP / (TP + FP)  # pos_pred_value
	recall = TP / (TP + FN)  # TP_rate/sensitivity
	false_discovery_rate = FP / (TP + FP)
	true_negative_rate = TN / (FP + TN)

	f1_score = hmean((precision, recall))

	# return {'Precision': precision, 'Recall': recall, 'True Negative Rate': true_negative_rate,
	#         'False Discovery Rate': false_discovery_rate, 'F1 Score': f1_score}
	return 'True Positive Rate: {:.2f}\nTrue Negative Rate: {:.2f}\nPrecision: {:.2f}\nFalse Discovery Rate: {:.2f}\nF1 ' \
	       'Score: {:.2f}\n'.format(recall, true_negative_rate, precision, false_discovery_rate, f1_score)


# %%


class AdaBoost:
	def __init__(self, n_estimators=None, base_estimator=None):
		if n_estimators is None:
			n_estimators = 5
		if base_estimator is None:
			base_estimator = DecisionTreeClassifier(max_depth=1)

		self.n_estimators = n_estimators
		self.classifier = base_estimator

		self.n_classes = 0
		self.classes = None
		self.class_binary_encoded = None

		self.estimators = []
		self.estimator_weights = []

	def fit(self, X, y, eps=1E-12):
		"""
		Only for binary classifier
		:param X: feature vectors
		:param y: target vector
		:param eps: precision to avoid division by 0
		"""
		N, _ = X.shape
		W = np.ones(N) / N

		self.classes = np.unique(y)
		self.n_classes = self.classes.size
		self.class_binary_encoded = OneHotEncoder(categories=self.classes, sparse=False)

		# y = copy.copy(y)
		# y[y == 0] = -1

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
		return classification_report(y, self.predict(X), target_names=['class 0', 'class 1'])


# %%


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
	@staticmethod
	def information_gain(x_parent, y_parent, split):
		y_child0 = y_parent[x_parent <= split]
		y_child1 = y_parent[x_parent > split]
		N = y_parent.shape[0]
		N_0 = y_child0.shape[0]
		if N_0 == 0 or N_0 == N:
			return 0
		p0 = float(N_0) / N
		p1 = 1 - p0
		return DtUtils.entropy(y_parent) - p0 * DtUtils.entropy(y_child0) - p1 * DtUtils.entropy(y_child1)

	@staticmethod
	def find_split(X, y, column):
		# Binarization -------------------
		sort_idx = np.argsort(X[:, column])
		x_values = X[sort_idx, column]
		y_values = y[sort_idx]
		split_indices = np.nonzero(y_values[1:] != y_values[:-1])[0]
		split_values = np.unique((x_values[split_indices] + x_values[split_indices + 1]) / 2)
		# --------------------------------

		max_ig, split_thresh = max(
			list(map(lambda split_value: (DtUtils.information_gain(x_values, y_values, split_value), split_value),
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
		self.left = None
		self.right = None
		self.prediction = collections.Counter(y).most_common()[0][0]

	def buildtree(self, X, y, depth, max_depth=None):
		# print(depth, end=' ')
		self.entropy = DtUtils.entropy(y)
		if len(y) == 1 or (max_depth is not None and depth >= max_depth) or self.entropy == 0:
			self.make_terminal_node(X, y)
			return

		best_feature, max_ig, best_split_thresh = max(
			list(map(lambda feature: ((feature,) + (DtUtils.find_split(X, y, feature))), range(X.shape[1]))),
			key=lambda v: v[1])

		if max_ig <= 0:
			self.make_terminal_node(X, y)
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
			return self.left.predict_val(x)
		else:
			return self.right.predict_val(x)

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

def run_adaboost(dataset_name, dataset, K=None):
	if K is None:
		K = [5, 10, 15, 20]
	X_train, X_test, y_train, y_test = dataset
	for k in K:
		model = AdaBoost(n_estimators=k, base_estimator=DecisionTree(max_depth=1))
		st_time = time.time()
		model.fit(X_train, y_train)
		en_time = time.time()
		print(dataset_name, ' AdaBoost x', k, ' Train:', '{:.4f}'.format(model.score(X_train, y_train)),
		      ' Test:', '{:.4f}'.format(model.score(X_test, y_test)))
		print('elaspled time: ', '{:.4f}'.format(en_time - st_time))
	print(flush=True)


def run_decisionTree(dataset_name, dataset):
	X_train, X_test, y_train, y_test = dataset
	model = DecisionTree()
	st_time = time.time()
	model.fit(X_train, y_train)
	en_time = time.time()
	print(dataset_name, ' Decision Tree', ' Train:', '{:.4f}'.format(model.score(X_train, y_train)),
	      ' Test:', '{:.4f}'.format(model.score(X_test, y_test)))
	print('Train:\n', model.report(X_train, y_train))
	print('Test:\n', model.report(X_test, y_test))
	print('elaspled time: ', '{:.4f}'.format(en_time - st_time))
	print(flush=True)


# %%
# from telco import train_test_dataset_telco

def preprocess_telco(df_orig):
	df = df_orig.copy()

	no_int_colms = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
	                'TechSupport', 'StreamingTV', 'StreamingMovies']

	for c in no_int_colms:
		df[c].replace(to_replace='No internet service', value='No', inplace=True)
	df['MultipleLines'].replace(to_replace='No phone service', value='No')

	# ',,,,tenure,Contract,,PaymentMethod,MonthlyCharges,TotalCharges,'

	yes_no_colms = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines'] \
	               + no_int_colms + ['PaperlessBilling', 'Churn']

	for c in yes_no_colms:
		df[c] = df[c].map(lambda s: 1 if s == 'Yes' else 0)

	df['gender'] = df['gender'].map(lambda s: 1 if s == 'Male' else 0)

	mult_val_colms = ['InternetService', 'Contract', 'PaymentMethod']
	for c in mult_val_colms:
		df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)
		df.drop(columns=[c], axis=1, inplace=True)

	df.TotalCharges.replace(to_replace=" ", value="", inplace=True)
	df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='ignore')
	df.TotalCharges.fillna(df.TotalCharges.mean(), inplace=True)

	return df


def train_test_dataset_telco():
	df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
	df.drop(columns=['customerID'], inplace=True)
	df = preprocess_telco(df_orig=df)

	X, y = df.drop(columns=['Churn']).to_numpy(), df['Churn'].to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# plt.figure(figsize = (24,20))
	# sns.heatmap(df.corr())

	return X_train, X_test, y_train, y_test


# %%

# from adult import train_test_dataset_adult
column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital - gain', 'capital - loss', 'hours - per - week',
                 'native - country', 'salary']

numeric_colms = ['age', 'fnlwgt', 'education-num', 'capital - gain', 'capital - loss', 'hours - per - week']
yes_no_colms = ['sex', 'salary']
cat_colms = [cat for cat in column_labels[:-1] if cat not in numeric_colms]
cat_colms = [cat for cat in cat_colms if cat not in yes_no_colms]


def setup_adult(df):
	df.columns = column_labels


"""
Both Train & Test
"""


def preprocess_adult(df, is_train=True, train_colms=None):
	for c in column_labels:
		df[c] = df[c].astype(str).str.strip()

	for c in cat_colms:
		df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)

	df.drop(columns=cat_colms, axis=1, inplace=True)

	df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0})
	if is_train:
		df['salary'] = df['salary'].replace({'>50K': 1, '<=50K': 0})
	else:
		df['salary'] = df['salary'].replace({'>50K.': 1, '<=50K.': 0})

	for c in numeric_colms:
		df[c] = df[c].replace(to_replace='?', value='')
		# df[c] = df[c].replace(to_replace='[ ]+', value='', regex=True)
		df[c] = pd.to_numeric(df[c], errors='ignore')
		df[c].fillna(value=df[c].mean(), inplace=True)

	if train_colms is None:
		df.sort_index(axis=1, inplace=True)
		train_colms = df.columns
	else:
		colms_missing = [c for c in train_colms if c not in df.columns]
		colms_extras = [c for c in df.columns if c not in train_colms]

		for colm in colms_missing:
			df[colm] = np.zeros(df.shape[0])

		for colm in colms_extras:
			df[colm] = np.zeros(df.shape[0])

		df.drop(columns=colms_extras, axis=1, inplace=True)
		df.sort_index(axis=1, inplace=True)
	return df, train_colms


def train_test_dataset_adult():
	df = pd.read_csv('data/adult/adult.data', header=None)
	df_test = pd.read_csv('data/adult/adult.test', skiprows=1, header=None)

	setup_adult(df)
	setup_adult(df_test)

	df_, cols = preprocess_adult(df)
	df_test_, _ = preprocess_adult(df_test, is_train=False, train_colms=cols)

	X_train, y_train = df_.drop(columns=['salary']).to_numpy(), df_['salary'].to_numpy()
	X_test, y_test = df_test_.drop(columns=['salary']).to_numpy(), df_test_['salary'].to_numpy()

	return X_train, X_test, y_train, y_test


# %%
# from credit import train_test_dataset_credit
def train_test_dataset_credit(n_neg_samples=20000):
	df = pd.read_csv('data/creditcard.csv')
	df.drop(columns=['Time'], axis=0, inplace=True)
	df_pos = df[df.Class == 1].reset_index()
	df.drop(df[df.Class == 1].index, inplace=True)
	df = df.sample(frac=1).reset_index(drop=True)

	df = pd.concat([df.head(n_neg_samples), df_pos], axis=0)
	df = df.sample(frac=1).reset_index(drop=True)
	df.drop(columns=['index'], axis=0, inplace=True)

	X, y = df.drop(columns=['Class']).to_numpy(), df['Class'].to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	return X_train, X_test, y_train, y_test


# %%
datasets = dict()
datasets['telco'] = train_test_dataset_telco()
datasets['adult'] = train_test_dataset_adult()
datasets['crdit'] = train_test_dataset_credit()

# %%
for (k, v) in datasets.items():
	run_decisionTree(k, v)

	run_adaboost(k, v)
