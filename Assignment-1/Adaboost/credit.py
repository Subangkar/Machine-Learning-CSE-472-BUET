import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split


def train_test_dataset_credit(n_neg_samples=20000, project_root='./', random_state=None):
	df = pd.read_csv(project_root + 'data/creditcard.csv')
	df.drop(columns=['Time'], axis=0, inplace=True)

	if n_neg_samples is not None:
		df_pos = df[df.Class == 1].reset_index()
		df.drop(df[df.Class == 1].index, inplace=True)
		df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

		df = pd.concat([df.head(n_neg_samples), df_pos], axis=0)
		df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
		df.drop(columns=['index'], axis=0, inplace=True)

	X, y = df.drop(columns=['Class']).to_numpy(), df['Class'].to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

	return X_train, X_test, y_train, y_test
