import seaborn as sns
import matplotlib.pyplot as plt

x = [15, 20, 25, 30, 35, 40, 45, 50, 100]
macro_svm = [0.4621, 0.5135, 0.5487, 0.5883, 0.6075, 0.6185, 0.6441, 0.6511, 0.7095]
micro_svm = [0.5259, 0.5667, 0.6122, 0.6421, 0.6624, 0.6798, 0.6997, 0.7079, 0.7686]
weighted_svm = [0.5489, 0.5834, 0.6262, 0.6511, 0.67115, 0.6861, 0.7047, 0.7122, 0.7686]

macro_keras = [0.4621, 0.5135, 0.5487, 0.5883, 0.6075, 0.6185, 0.6441, 0.6511, 0.7095]
micro_keras = [0.5259, 0.5667, 0.6122, 0.6421, 0.6624, 0.6798, 0.6997, 0.7079, 0.7686]
weighted_keras = [0.5489, 0.5834, 0.6262, 0.6511, 0.67115, 0.6861, 0.7047, 0.7122, 0.7686]

plt.plot(x,macro_svm, marker = 'o', color='#f70d1a', label="Macro f1")
plt.plot(x,micro_svm, marker = 'x', color='#08088A', label="Micro f1")
plt.plot(x,weighted_svm, marker = '+', color='#FFBF00', label="weighted f1", linestyle = '--')
plt.xlabel("#Neurons")
plt.xticks(x)
plt.ylabel("F1-Scores")
plt.ylim([0, 1])
plt.legend(loc="upper left")
plt.show()
