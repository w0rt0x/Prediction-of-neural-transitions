import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bar_charts(data: dict, path:str, bar_width: float=0.05):
    x = np.arange(4)
    acc = [34, 56, 12, 89]
    acc_r = [1, 1, 1, 1]
    f1_00 = [34, 56, 12, 89]
    f1_00r = [1, 1, 1, 1]
    f1_01 = [12, 56, 78, 45]
    f1_01r = [1, 1, 1, 1]
    f1_10 = [12, 56, 78, 45]
    f1_10r = [1, 1, 1, 1]
    f1_11 = [12, 56, 78, 45]
    f1_11r = [1, 1, 1, 1]
    
    # Accuracy
    plt.bar(x-0.25, acc, bar_width, color='black', label="Accuracy")
    plt.bar(x-0.2, acc_r, bar_width, color='grey')
    # f1 Scores
    plt.bar(x-0.15, f1_00, bar_width, color='cyan')
    plt.bar(x-0.05, f1_00r, bar_width, color='grey')

    plt.bar(x, f1_01, bar_width, color='green')
    plt.bar(x+0.05, f1_01r, bar_width, color='grey')

    plt.bar(x+0.1, f1_10, bar_width, color='red')
    plt.bar(x+0.15, f1_10r, bar_width, color='grey')

    plt.bar(x+0.2, f1_11, bar_width, color='yellow')
    plt.bar(x+0.25, f1_11r, bar_width, color='grey')

    plt.xticks(x, ['PCA', 't-SNE', 'uMAP', 'n most\n active neurons'])
    plt.ylabel("Scores")
    plt.xlabel("Dimensionality Reduction Method")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get data 
    # plot data
    d = {"PCA": {"Accuracy": 0.1, "f1": 0.2}, "t-SNE": {"Accuracy": 0.4, "f1": 0.5}}
    bar_charts(d, "str")

# Categorien(SVM):
# Dimesion Methode
# 1 oder mehrere Vektoren

# Zu zeigen:
# Random mit allem!!!, weighted f1, Accuracy, macro, f1 für jede einzelne Klasse