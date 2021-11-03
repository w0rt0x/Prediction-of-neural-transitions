#from Classifier import SVMclassifier
#from Plotter import Plotter, get_all_pop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


p1 = pd.read_csv(r'C:\Users\Sam\Desktop\1days.csv')
p2 = pd.read_csv(r'C:\Users\Sam\Desktop\2days.csv')


names = list(p1["model"])
ins = []
for i in range(len(names)):
    if "shuffled" in names[i]:
        ins.append("transitions to next day\n with shuffled labels")
    else:
        ins.append("transitions to next day")
p1["transition"] = ins

names = list(p2["model"])
ins = []
for i in range(len(names)):
    if "shuffled" in names[i]:
        ins.append("transitions to day\n after the next day\n with shuffled labels")
    else:
        ins.append("transitions to day\n after the next day")
p2["transition"] = ins


df = pd.concat([p1, p2])

sns.set_theme(palette="pastel")
sns.set(font_scale=1.7)
sns.boxplot(x="transition", y="macro F1 score", data=df)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.title("Prediction performance of transitions across one day and across two days")
plt.show()


