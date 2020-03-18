# %%
import random
import sys
import time

# %%
random.seed(1)
# sys.stdout = open('report.log', 'w')

# %%
from models.adaboost import AdaBoost

# %%
from models.decisiontree import DecisionTree


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
from telco import train_test_dataset_telco
from adult import train_test_dataset_adult
from credit import train_test_dataset_credit

# %%
datasets = dict()
datasets['telco'] = train_test_dataset_telco()
datasets['adult'] = train_test_dataset_adult()
datasets['crdit'] = train_test_dataset_credit()

# %%
for (k, v) in datasets.items():
	run_decisionTree(k, v)

	run_adaboost(k, v)
