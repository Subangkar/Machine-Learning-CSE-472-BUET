import pandas as pd
import plotly.offline as py  # visualization
from sklearn.model_selection import train_test_split

from utils import countplot

py.init_notebook_mode(connected=True)  # visualization
import plotly.graph_objs as go  # visualization

from models.adaboost import AdaBoost
from models.decisiontree import DecisionTree


# function  for pie plot for customer attrition types
def plot_pie(churn, not_churn, column):
	trace1 = go.Pie(values=churn[column].value_counts().values.tolist(),
	                labels=churn[column].value_counts().keys().tolist(),
	                hoverinfo="label+percent+name",
	                domain=dict(x=[0, .48]),
	                # name="Churn Customers",
	                marker=dict(line=dict(width=2, color="rgb(243,243,243)")),
	                hole=.6
	                )
	trace2 = go.Pie(values=not_churn[column].value_counts().values.tolist(),
	                labels=not_churn[column].value_counts().keys().tolist(),
	                hoverinfo="label+percent+name",
	                domain=dict(x=[.52, 1]),
	                # name="Non churn customers",
	                marker=dict(line=dict(width=2, color="rgb(143,143,143)")),
	                hole=.6
	                )

	layout = go.Layout(dict(title=column + " distribution in customer attrition ",
	                        plot_bgcolor="rgb(243,243,243)",
	                        paper_bgcolor="rgb(243,243,243)",
	                        annotations=[dict(text="churn customers",
	                                          font=dict(size=13),
	                                          showarrow=False,
	                                          x=.15, y=.5),
	                                     dict(text="Non churn customers",
	                                          font=dict(size=13),
	                                          showarrow=False,
	                                          x=.88, y=.5
	                                          )
	                                     ]
	                        )
	                   )
	data = [trace1, trace2]
	fig = go.Figure(data=data, layout=layout)
	fig.update_layout(
		autosize=False,
		width=500,
		height=400)
	py.iplot(fig)


# function  for histogram for customer attrition types
def histogram(churn, not_churn, column):
	trace1 = go.Histogram(x=churn[column],
	                      histnorm="percent",
	                      name="Churn Customers",
	                      marker=dict(line=dict(width=.5,
	                                            color="black")),
	                      opacity=.9)

	trace2 = go.Histogram(x=not_churn[column],
	                      histnorm="percent",
	                      name="Non churn customers",
	                      marker=dict(line=dict(width=.5,
	                                            color="black")),
	                      opacity=.9)

	data = [trace1, trace2]
	layout = go.Layout(dict(title=column + " distribution in customer attrition ",
	                        plot_bgcolor="rgb(243,243,243)",
	                        paper_bgcolor="rgb(243,243,243)",
	                        xaxis=dict(gridcolor='rgb(255, 255, 255)',
	                                   title=column,
	                                   zerolinewidth=1,
	                                   ticklen=5,
	                                   gridwidth=2
	                                   ),
	                        yaxis=dict(gridcolor='rgb(255, 255, 255)',
	                                   title="percent",
	                                   zerolinewidth=1,
	                                   ticklen=5,
	                                   gridwidth=2
	                                   ),
	                        )
	                   )
	fig = go.Figure(data=data, layout=layout)

	py.iplot(fig)


# function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df):
	df = df.sort_values(by="Churn", ascending=True)
	classes = df["Churn"].unique().tolist()

	class_code = {classes[k]: k for k in range(2)}

	color_vals = [class_code[cl] for cl in df["Churn"]]

	pl_colorscale = "Portland"

	text = [df.loc[k, "Churn"] for k in range(len(df))]

	trace = go.Splom(dimensions=[dict(label="tenure",
	                                  values=df["tenure"]),
	                             dict(label='MonthlyCharges',
	                                  values=df['MonthlyCharges']),
	                             dict(label='TotalCharges',
	                                  values=df['TotalCharges'])],
	                 text=text,
	                 marker=dict(color=color_vals,
	                             colorscale=pl_colorscale,
	                             size=3,
	                             showscale=False,
	                             line=dict(width=.1,
	                                       color='rgb(230,230,230)'
	                                       )
	                             )
	                 )
	axis = dict(showline=True,
	            zeroline=False,
	            gridcolor="#fff",
	            ticklen=4
	            )

	layout = go.Layout(dict(title="Scatter plot matrix for Numerical columns for customer attrition",
	                        autosize=False,
	                        height=800,
	                        width=800,
	                        dragmode="select",
	                        hovermode="closest",
	                        plot_bgcolor='rgba(240,240,240, 0.95)',
	                        xaxis1=dict(axis),
	                        yaxis1=dict(axis),
	                        xaxis2=dict(axis),
	                        yaxis2=dict(axis),
	                        xaxis3=dict(axis),
	                        yaxis3=dict(axis),
	                        )
	                   )
	data = [trace]
	fig = go.Figure(data=data, layout=layout)
	py.iplot(fig)


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


def df_dist_details(df):
	for c in df.columns:
		print(c)
		for v in df['Churn'].unique():
			print('< ' + str(v) + ' >')
			print(df[c][df['Churn'] == v].value_counts())
		print(end='\n\n')


if __name__ == '__main__':
	X_train, X_test, y_train, y_test = train_test_dataset_telco()

	df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

	target_col = ["Churn"]
	cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()
	cat_cols = [x for x in cat_cols if x not in target_col]
	countplot(df, "Churn", cat_cols[:-4], n_max=4)

	dtc = DecisionTree()
	dtc.fit(X_train, y_train)
	print('Decision Tree: ', dtc.score(X_test, y_test))

	model = AdaBoost(n_estimators=15, base_estimator=DecisionTree(max_depth=1))
	model.fit(X_train, y_train)
	print('AdaBoost : ', model.score(X_test, y_test))
