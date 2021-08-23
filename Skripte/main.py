#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv(r'C:\Users\Sam\Desktop\model_cpmparison_40.csv')
df2 = pd.read_csv(r'C:\Users\Sam\Desktop\model_cpmparison_PCA.csv')
df3 = pd.read_csv(r'C:\Users\Sam\Desktop\model_cpmparison_tSNE.csv')

df = pd.concat([df1, df2, df3])
sns.set_theme(palette="pastel")
sns.boxplot(x="model", y="macro F1 score", hue="input", data=df)
plt.title("Results of population-wise 5-fold Cross Validation with different SVM-Kernels and inputs")
plt.show()


