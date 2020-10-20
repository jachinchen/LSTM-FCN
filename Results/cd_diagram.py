import Orange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gmean

data = pd.read_csv('train_loss_old_datasets.csv')
df = data.iloc[:, 1:]

number_of_classifers = df.shape[1]

mean_values = np.array(df.rank(1, method='min', ascending=False).mean(0))
geo_mean_lstm = gmean(df.rank(1, method='min', ascending=False).iloc[:, 1])
geo_mean_alstm = gmean(df.rank(1, method='min', ascending=False).iloc[:, 0])
names = df.columns.values
cd = Orange.evaluation.compute_CD(mean_values, df.shape[0])  # tested on 30 datasets
Orange.evaluation.graph_ranks(mean_values, names, cd=cd, width=7, textspace=1.5)
plt.show()
