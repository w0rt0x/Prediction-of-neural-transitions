import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

dataframe = pd.read_csv(r'C:\Users\Sam\Desktop\grid1Pop.csv')
means = dataframe['mean_test_score']
stds = dataframe['std_test_score']
params = dataframe['params']


for i in range(len(means)):
    #print(means[i], stds[i], params[i])
    p = ast.literal_eval(params[i])
    c = p['C']
    g = p['gamma']
    print(c, g, means[i])

x = np.empty((13, 13))
for i in range(13):
    for j in range(13):
        x[i][j] = round(means[i * 13 + j], 4)

c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000, 10000]
y = deepcopy(c)
y.reverse()
ax = sns.heatmap(x, annot=True, vmin=0, vmax=1, xticklabels=c, yticklabels=c)
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title("Grid Search for bl693_no_white_Pop05:\n 'kernel':['rbf'], 'C':c, 'gamma': c, 'class_weight':['balanced']")
plt.show()
