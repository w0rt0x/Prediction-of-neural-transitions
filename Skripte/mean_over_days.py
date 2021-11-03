import pandas as pd
from getter_for_populations import get_all_pop, sort_out_populations
import seaborn as sns
import matplotlib.pyplot as plt


ok, not_ok = sort_out_populations()
path = r'D:\Dataframes\single_values\std'

full_df = pd.DataFrame(columns=[ 'day', 'std', 'transition'])

c = 0
for pop in ok:
    df = pd.read_csv(path + "\\" + pop + ".csv")
    for index, row in df.iterrows():
        if int(row['label'][1]) == 4:
            pass
        else:
            ls = [row['label'][1], row['Component 1'], row['response']]
            full_df.loc[c] = ls
            c +=1

print(full_df)

sns.set_theme(palette="pastel")
sns.set(font_scale=1.7)
sns.boxplot(x="transition", y="std", hue="day", data=full_df, showfliers=False)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Shift of std over the four days (No outliers)")
plt.show()


### GLEICHES MIT STD