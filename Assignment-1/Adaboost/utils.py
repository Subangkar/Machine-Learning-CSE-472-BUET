from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns  # visualization
import matplotlib.pyplot as plt  # visualization
from scipy.stats import hmean


def perf_metrics_2X2(y_true, y_pred):
	# TP = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
	# TN = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
	# FN = np.sum(y_true[y_true == 1] == y_pred[y_true == 0])
	# FP = np.sum(y_true[y_true == 0] == y_pred[y_true == 1])
	#
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
	return 'True Positive Rate: {:.1f}\nTrue Negative Rate: {:.1f}\nPrecision: {:.1f}\nFalse Discovery Rate: {:.1f}\nF1 ' \
	       'Score: {:.1f}\n'.format(recall, true_negative_rate, precision, false_discovery_rate, f1_score)


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
