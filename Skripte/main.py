#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv(r'C:\Users\Sam\Desktop\2days.csv')
df2 = pd.read_csv(r'C:\Users\Sam\Desktop\1days.csv')
df = pd.concat([df1, df2])
sns.set_theme(palette="pastel")
sns.boxplot(x="model", y="macro F1 score", data=df)
plt.title("Population-wise 5-fold Cross Validation with the SVM (rbf-Kernel, C=1, gamma=0.5, balanced class weights)\nwith SMOTE used on training-data: Using the 40 most active neurons as input")
plt.show()


