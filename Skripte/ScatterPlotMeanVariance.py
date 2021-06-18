import matplotlib.pyplot as plt
import numpy as np 
from get_data_for_DL import getMeanAndVariance
from copy import deepcopy

X, x, Y, y = getMeanAndVariance(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\100_Transition_multiclass')

X = np.concatenate((X, x))
Y = np.concatenate((Y, y))

d = dict()
for i in range(len(Y)):
    if Y[i] in d:
        l = deepcopy(d[Y[i]])
        l.append(X[i])
        d[Y[i]] = l
    else:
        d[Y[i]] = [X[i]]

for key in list(d.keys()):
    d[key] = np.asarray(d[key])

    plt.plot(d[key], marker='.', linestyle='none', markersize=7, label=key)

plt.ylabel('Variance')
plt.xlabel('Mean')
plt.legend()
plt.show()

"""
for i in range(len(X)):

    plt.plot(X[i], marker='.', linestyle='none', markersize=7, label=Y[i])

plt.ylabel('Variance')
plt.xlabel('Mean')
plt.legend()
plt.show()
"""