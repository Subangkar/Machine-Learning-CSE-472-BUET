# %%
import pandas as pd
import numpy as np
import seaborn as sn

# %%
from sklearn.metrics import confusion_matrix

column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital - gain', 'capital - loss', 'hours - per - week',
                 'native - country', 'salary']

numeric_colms = ['age', 'fnlwgt', 'education-num', 'capital - gain', 'capital - loss', 'hours - per - week']
yes_no_colms = ['sex', 'salary']
cat_colms = [cat for cat in column_labels[:-1] if cat not in numeric_colms]
cat_colms = [cat for cat in cat_colms if cat not in yes_no_colms]

"""
0  age: continuous.
1  workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
2  fnlwgt: continuous.
3  education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
4  education-num: continuous.
5  marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
6  occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
7  relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
8  race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
9  sex: Female, Male.
10 capital-gain: continuous.
11 capital-loss: continuous.
12 hours-per-week: continuous.
13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

14 salary: <=50K, >50K
"""


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


def dataframes_adult(df, df_test):
	setup_adult(df)
	setup_adult(df_test)

	return df, df_test


def train_test_dataset(df, df_test):
	df_, cols = preprocess_adult(df)
	df_test_, _ = preprocess_adult(df_test, is_train=False, train_colms=cols)

	X_train, y_train = df_.drop(columns=['salary']).to_numpy(), df_['salary'].to_numpy()
	X_test, y_test = df_test_.drop(columns=['salary']).to_numpy(), df_test_['salary'].to_numpy()

	return X_train, X_test, y_train, y_test


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


if __name__ == '__main__':
	df = pd.read_csv('../data/adult/adult.data', header=None)
	df_test = pd.read_csv('../data/adult/adult.test', skiprows=1, header=None)

	df, df_test = dataframes_adult(df, df_test)

	X_train, X_test, y_train, y_test = train_test_dataset(df, df_test)

	from models.decisiontree import DecisionTree

	dtc = DecisionTree()
	dtc.fit(X_train, y_train)
	sn.heatmap(confusion_matrix(y_test, dtc.predict(X_test)), annot=True, annot_kws={"size": 32})
	print(dtc.score(X_test, y_test))

	D = 1
	K = 15

	from models.adaboost import AdaBoost

	model = AdaBoost(n_estimators=K, base_estimator=DecisionTree(max_depth=1))
	model.fit(X_train, y_train)
	sn.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, annot_kws={"size": 32})
	print(model.score(X_test, y_test))
