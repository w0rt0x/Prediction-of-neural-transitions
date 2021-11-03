import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

"""
arr = np.array([[0.73,  0.14, 0.08, 0.03], [0.32, 0.47, 0.10, 0.09], [0.18, 0.11, 0.56, 0.14], [0.09, 0.10, 0.13, 0.66]])

df_cm = pd.DataFrame(arr, index = [i for i in ['0->0', '0->1', '1->0', '1->1']],
                  columns = [i for i in ['0->0', '0->1', '1->0', '1->1']])

sns.set(font_scale=1.5)
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar_kws={'label': 'percentage of labels in particular class'}, vmin=0, vmax=1)
plt.title("Prediction results of SVM with rbf-kernel")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
"""
#https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot

arr = np.array([[0.73,  0.14, 0.08, 0.03], [0.32, 0.47, 0.1, 0.09], [0.18, 0.11, 0.56, 0.14], [0.09, 0.1, 0.13, 0.66]])



df_cm = pd.DataFrame(arr, index = [i for i in ['0->0', '0->1', '1->0', '1->1']],
                  columns = [i for i in ['0->0', '0->1', '1->0', '1->1']])

sns.set(font_scale=1.3)
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar_kws={'label': 'row-wise percentage'}, vmin=0, vmax=1)
plt.title("Prediction results of SVM with rbf-kernel")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()



df = pd.DataFrame([[0.2961, 0.9585]], columns=["2", "20"])
sns.set(font_scale=1.8)
sns.barplot(data=df, orient = 'h')
plt.xlabel("total variance of components")
plt.ylabel("Number of principle components")
plt.title("Used principle components and their explained variance")
plt.show()
