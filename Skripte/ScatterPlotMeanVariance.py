import matplotlib.pyplot as plt
import numpy as np 
from get_data_for_DL import getMeanAndVariance
from copy import deepcopy

X, Y = getMeanAndVariance(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\100_Transition_multiclass')

x = []
y = []
table = {"0->0":0, "0->1":1, "1->0":2, "1->1":3}
for i in range(len(X)):
    x.append(X[i][0])
    y.append(Y[i])

plt.scatter(x, y)

plt.ylabel('Transitions')
plt.xlabel('Mean Acticity')
#ax.set_xticks(["0->0", "0->1", "1->0","1->1"])
plt.axvline(x=0.4, color='grey')
plt.show()

"""
d = dict()
for i in range(len(Y)):
    if Y[i] in d:
        l = deepcopy(d[Y[i]])
        l.append(X[i])
        d[Y[i]] = l
    else:
        d[Y[i]] = [X[i]]

table = {"0->0":0, "0->1":1, "1->0":2, "1->1":3}
for key in list(d.keys()):
    d[key] = np.asarray(d[key])
"""
   


"""
plt.plot(d[key], label=key)

plt.ylabel('Variance')
plt.xlabel('Mean')
plt.legend()
plt.axvline(x=0.4, color='grey')
plt.show()
"""
"""
for i in range(len(X)):

    plt.plot(X[i], marker='.', linestyle='none', markersize=7, label=Y[i])

plt.ylabel('Variance')
plt.xlabel('Mean')
plt.legend()
plt.show()
"""