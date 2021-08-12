import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

print(tips)
# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="size",
            data=tips)

plt.show()


#https://seaborn.pydata.org/examples/grouped_boxplot.html
#https://www.geeksforgeeks.org/grouped-boxplots-in-python-with-seaborn/