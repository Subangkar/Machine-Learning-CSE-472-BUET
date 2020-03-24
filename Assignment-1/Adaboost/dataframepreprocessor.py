import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


class DataFramePreprocessor:
	def __init__(self, df_train, drop_colms=None):
		if drop_colms is None:
			drop_colms = []
		self.label_encoder = LabelEncoder()
		self.one_hot_encoder = OneHotEncoder()
		self.columns_sorted = sorted([c for c in df_train.columns if c not in drop_colms])

	def trim_whitespace(self, df, columns=None):
		return DataFramePreprocessor.trim_wsp_util(df, columns=columns, inplace=True)

	def process_multi_valued_cat(self, df, multi_categorical_columns=None):
		return DataFramePreprocessor.binarization_util(df, categorical_columns=multi_categorical_columns)

	def process_numeric(self, df, numeric_columns=None):
		return DataFramePreprocessor.replace_numeric_na_util(df, numeric_columns=numeric_columns, inplace=True)

	def preprocess_test(self, df, mul_cat_colms=None, num_colms=None, bin_cat_colms=None, target_colm=None,
	                    bin_replace_map=None, num_na_symbol='', drop_colms=None):
		return self.preprocess(df, mul_cat_colms=mul_cat_colms, num_colms=num_colms, bin_cat_colms=bin_cat_colms,
		                target_colm=target_colm, bin_replace_map=bin_replace_map, num_na_symbol=num_na_symbol,
		                drop_colms=drop_colms, is_test=True, sort_index=True)

	def preprocess_train(self, df, mul_cat_colms=None, num_colms=None, bin_cat_colms=None, target_colm=None,
	                     bin_replace_map=None, num_na_symbol='', drop_colms=None):
		return self.preprocess(df, mul_cat_colms=mul_cat_colms, num_colms=num_colms, bin_cat_colms=bin_cat_colms,
		                target_colm=target_colm, bin_replace_map=bin_replace_map, num_na_symbol=num_na_symbol,
		                drop_colms=drop_colms, is_train=True, sort_index=True)

	def preprocess(self, df, mul_cat_colms=None, num_colms=None, bin_cat_colms=None, target_colm=None,
	               bin_replace_map=None, num_na_symbol='', drop_colms=None, is_train=False, is_test=False,
	               sort_index=False):
		"""

		:param df:
		:param mul_cat_colms:
		:param num_colms:
		:param bin_cat_colms:
		:param target_colm:
		:param bin_replace_map:
		:param num_na_symbol:
		:param drop_colms:
		:param is_train: set True if Train/Test pre-processing are different and df is train set
		:param is_test:  set True if Train/Test pre-processing are different and df is test set
		:param sort_index:set True if Train/Test pre-processing are different
		:return: preprocessed dataframe
		"""

		if bin_replace_map is None:
			bin_replace_map = []
		if bin_cat_colms is None:
			bin_cat_colms = []
		if num_colms is None:
			num_colms = []
		if mul_cat_colms is None:
			mul_cat_colms = []
		if drop_colms is None:
			drop_colms = []

		df.drop(columns=drop_colms, axis=1, inplace=True, errors='ignore')

		df = DataFramePreprocessor.trim_wsp_util(df)
		df = DataFramePreprocessor.binary_replace_util(df,
		                                               categorical_columns=bin_cat_colms,
		                                               replace_map=bin_replace_map)
		df = DataFramePreprocessor.binarization_util(df, categorical_columns=mul_cat_colms)
		df = DataFramePreprocessor.replace_numeric_na_util(df, numeric_columns=num_colms, na_symbol=num_na_symbol)

		if is_train and is_test:
			raise Exception('Invalid selection')
		elif is_train:
			self.columns_sorted = sorted(df.columns)
		elif is_test:
			DataFramePreprocessor.align_test_colms_util(df, self.columns_sorted)

		if sort_index:
			df.sort_index(axis=1, inplace=True)

		return df

	@staticmethod
	def trim_wsp_util(df_orig, columns=None, inplace=True):
		if not inplace:
			df = df_orig.copy(deep=True)
		else:
			df = df_orig

		if columns is None:
			columns = df.columns

		# for c in columns:
		for c in df.select_dtypes(include=['object']).columns:
			df[c] = df[c].astype(str).str.strip()

		return df

	# Out place
	@staticmethod
	def binarization_util(df, categorical_columns=None):
		"""
		:param df:
		:param categorical_columns: array of categorical columns
		:param inplace:
		:return: preprocessed df
		"""

		for c in categorical_columns:
			df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)

		df.drop(columns=categorical_columns, axis=1, inplace=True, errors='ignore')

		return df

	@staticmethod
	def replace_numeric_na_util(df_orig, numeric_columns=None, inplace=True, na_symbol=''):
		"""
		:param df_orig:
		:param numeric_columns: array of numeric columns
		:param na_symbol: string value which denotes that value is missing
		:param inplace:
		:return: preprocessed df
		"""
		if not inplace:
			df = df_orig.copy(deep=True)
		else:
			df = df_orig

		for c in numeric_columns:
			df[c] = df[c].replace(to_replace=na_symbol, value='')
			# df[c] = df[c].replace(to_replace='[ ]+', value='', regex=True)
			df[c] = pd.to_numeric(df[c], errors='ignore')
			df[c].fillna(value=df[c].mean(), inplace=True)

		return df

	@staticmethod
	def binary_replace_util(df_orig, categorical_columns=None, replace_map=None, inplace=True):
		"""
		:param df_orig:
		:param categorical_columns: array of categorical columns
		:param replace_map: categorical_columns's corresponding array of map
		:param inplace:
		:return: preprocessed df
		"""
		if len(categorical_columns) != len(replace_map):
			raise Exception('Binary cat colms & map len mismatch')

		if not inplace:
			df = df_orig.copy(deep=True)
		else:
			df = df_orig

		for c, m in zip(categorical_columns, replace_map):
			df[c].replace(m, inplace=inplace)

		return df

	@staticmethod
	def align_test_colms_util(df, train_colms):
		test_colms = df.columns
		colms_missing = [c for c in train_colms if c not in test_colms]
		colms_extras = [c for c in test_colms if c not in train_colms]
		for colm in colms_missing:
			df[colm] = np.zeros(df.shape[0])
		df.drop(columns=colms_extras, axis=1, inplace=True, errors='ignore')
