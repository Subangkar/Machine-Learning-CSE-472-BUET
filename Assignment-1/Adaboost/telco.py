# Importing libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt  # visualization
# from PIL import Image
# %matplotlib inline
import pandas as pd
import seaborn as sns  # visualization
import itertools
import warnings

warnings.filterwarnings("ignore")
import io
import plotly.offline as py  # visualization

py.init_notebook_mode(connected=True)  # visualization
import plotly.graph_objs as go  # visualization
import plotly.tools as tls  # visualization
import plotly.figure_factory as ff  # visualization

import seaborn as sns


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
	classes

	class_code = {classes[k]: k for k in range(2)}
	class_code

	color_vals = [class_code[cl] for cl in df["Churn"]]
	color_vals

	pl_colorscale = "Portland"

	pl_colorscale

	text = [df.loc[k, "Churn"] for k in range(len(df))]
	text

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

	# %% md
	# Numeric
	# %%
	df.TotalCharges.replace(to_replace=" ", value="", inplace=True)
	df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='ignore')
	df.TotalCharges.fillna(df.TotalCharges.mean(), inplace=True)

	return df