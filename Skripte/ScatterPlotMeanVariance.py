import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

path = r'D:\Dataframes\isomap_2d'
pop = 'bl693_no_white_Pop06'
df = pd.read_csv(path + '\\' + pop + '.csv')
header = df['label'].tolist()
x = df['PC1'].tolist()
y = df['PC2'].tolist()
label = df['response'].tolist()
i = header.index('(4, 1)')
x = x[:i]
y = y[:i]
label = label[:i]

"""
table = {"0->0":0, "0->1":1, "1->0":2, "1->1":3}
for i in range(len(label)):
    cols.append(table[label[i]])
"""

plt.scatter(x, y, c=label, cmap='plasma')

plt.ylabel('ISOMAP Component 1')
plt.xlabel('ISOMAP Component 1')
plt.legend(label)
plt.show()

