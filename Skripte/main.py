#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv(r'C:\Users\Sam\Desktop\FFN_pop-wise.csv')
df2 = pd.read_csv(r'C:\Users\Sam\Desktop\FFN_across-pop.csv')
df3 = pd.read_csv(r'C:\Users\Sam\Desktop\FFN_day3.csv')

df = pd.concat([df1, df2, df3])
sns.set_theme(palette="pastel")
sns.boxplot(x="model", y="macro F1 score", hue="input", data=df)
plt.title("Prediction results using the FFN and the 40 most active neurons")
plt.show()


