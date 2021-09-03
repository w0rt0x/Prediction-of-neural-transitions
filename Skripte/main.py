#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from getter_for_populations import sort_out_populations


#df1 = pd.read_csv(r'C:\Users\Sam\Desktop\2days.csv')
#df2 = pd.read_csv(r'C:\Users\Sam\Desktop\1days.csv')
ok, not_ok = sort_out_populations()
dfs = []
for pop in ok:
    path = r'D:\Dataframes\single_values\mean_over_all\{}.csv'.format(pop)
    dfs.append(pd.read_csv(path))
df = pd.concat(dfs)
df = df[df.response != "0"]
sns.set_theme(palette="pastel")
sns.boxplot(x="response", y="Component 1", data=df, showfliers = False, order=["0->0", "0->1", "1->0", "1->1"]).set(
    xlabel='Labels', 
    ylabel='mean neural activity'
)
plt.title("mean neural activity of trials (without outliers)")
plt.show()


