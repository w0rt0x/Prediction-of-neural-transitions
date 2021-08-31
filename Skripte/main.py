#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Sam\Desktop\std_and_mean_comparison.csv')
sns.set_theme(palette="pastel")
sns.boxplot(x="model", y="macro F1 score", data=df)
plt.title("Population-wise 5-fold Cross Validation with the SVM (linear Kernel, C=1, balanced class weights)\nwith SMOTE used on training-data: Using mean, standard-deviation (std) and both as input")
plt.show()


