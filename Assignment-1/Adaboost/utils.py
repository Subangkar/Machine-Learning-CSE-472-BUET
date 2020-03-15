from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns  # visualization
import matplotlib.pyplot as plt  # visualization


def label_encode(df):
	df_c = df.copy()
	for column in df.columns:
		if df[column].dtype != np.number:
			df_c[column] = LabelEncoder().fit_transform(df[column])
	return df_c


def countplot(df, target, bases, n_max):
	import math
	i = 1
	if n_max is None:
		n_max = math.ceil(math.sqrt(len(bases)))
	else:
		n_max = math.ceil(math.sqrt(n_max))
	for base in bases:
		plt.subplot(n_max, n_max, i)
		sns.countplot(df[target], hue=df[base])
		i += 1
		if i > n_max * n_max:
			break
