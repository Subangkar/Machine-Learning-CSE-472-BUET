# %%

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

RANDOM_STATE = 15
# sys.stdout = open('report.log', 'w')

# %%

from models.adaboost import AdaBoost

# %%

from models.decisiontree import DecisionTree


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

def run_adaboost(dataset_name, dataset, K=None, random_state=None):
	if K is None:
		K = [5, 10, 15, 20]
	X_train, X_test, y_train, y_test = dataset
	for k in K:
		model = AdaBoost(n_estimators=k, base_estimator=DecisionTree(max_depth=1))
		st_time = time.time()
		model.fit(X_train, y_train, random_state=random_state)
		en_time = time.time()
		print(dataset_name, ' AdaBoost x', k, ' Train:', '{:.4f}'.format(model.score(X_train, y_train)),
		      ' Test:', '{:.4f}'.format(model.score(X_test, y_test)))
		model.plot_cm(X_test, y_test)
		print('elaspled time: ', '{:.4f}'.format(en_time - st_time))
	print(flush=True)


def run_decisionTree(dataset_name, dataset, random_state=None):
	X_train, X_test, y_train, y_test = dataset
	model = DecisionTree()
	st_time = time.time()
	model.fit(X_train, y_train, random_state=random_state)
	en_time = time.time()
	print(dataset_name, ' Decision Tree', ' Train:', '{:.4f}'.format(model.score(X_train, y_train)),
	      ' Test:', '{:.4f}'.format(model.score(X_test, y_test)))
	print('Train:\n' + model.report(X_train, y_train))
	print('Test:\n' + model.report(X_test, y_test))
	model.plot_cm(X_test, y_test)
	print('elaspled time: ', '{:.4f}'.format(en_time - st_time))
	print(flush=True)


# %%

from telco import train_test_dataset_telco
from adult import train_test_dataset_adult
from credit import train_test_dataset_credit

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

datasets = dict()
datasets['telco'] = train_test_dataset_telco(project_root='./', random_state=RANDOM_STATE)
datasets['adult'] = train_test_dataset_adult(project_root='./', random_state=RANDOM_STATE)
datasets['crdit'] = train_test_dataset_credit(project_root='./', random_state=RANDOM_STATE)

# %%

for (k, v) in datasets.items():
	run_decisionTree(k, v, random_state=RANDOM_STATE)

	run_adaboost(k, v, random_state=RANDOM_STATE)
