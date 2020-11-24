# %%

import sys
import time

# %%

RANDOM_STATE = 15
# sys.stdout = open('report.log', 'w')

# %%

from models.adaboost import AdaBoost

# %%

from models.decisiontree import DecisionTree


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

# %%

datasets = dict()
datasets['telco'] = train_test_dataset_telco(project_root='./', random_state=RANDOM_STATE)
datasets['adult'] = train_test_dataset_adult(project_root='./', random_state=RANDOM_STATE)
datasets['crdit'] = train_test_dataset_credit(project_root='./', random_state=RANDOM_STATE)

# %%

for (k, v) in datasets.items():
	run_decisionTree(k, v, random_state=RANDOM_STATE)

	run_adaboost(k, v, random_state=RANDOM_STATE)
