import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

path = r'C:\Users\Sam\Desktop'
pop = 'bl693_no_white_Pop05'
df = pd.read_csv(path + '\\' + pop + '.csv')
header = df['label'].tolist()
x = df['Component 1'].tolist()
y = df['Component 2'].tolist()
label = df['response'].tolist()
i = header.index('(4, 1)')
x = x[:i]
y = y[:i]
label = label[:i]

cols=[]
table = {"0->0":0, "0->1":1, "1->0":2, "1->1":3}
for i in range(len(label)):
    cols.append(table[label[i]])


plt.scatter(x, y, c=cols, cmap='plasma')

plt.ylabel('ISOMAP Component 1')
plt.xlabel('ISOMAP Component 1')
plt.legend(label)
plt.show()

