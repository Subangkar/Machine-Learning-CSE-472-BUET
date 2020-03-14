from sklearn.preprocessing import LabelEncoder
import numpy as np


def label_encode(df):
	df_c=df.copy()
	for column in df.columns:
		if df[column].dtype != np.number:
			df_c[column] = LabelEncoder().fit_transform(df[column])
	return df_c
