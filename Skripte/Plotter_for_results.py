from getter_for_populations import sort_out_populations
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import os
from os.path import isfile, join
from collections import Counter
from copy import deepcopy
from data_holder import Data
from Classifier import SVMclassifier
from DeepLearning import FeedforwardNetWork


class PerformancePlotter:
    """
    Plotter for comparing models and their performance
    """
    def __init__(self, populations: list, path:str):
        """
        Needs List of populations and directory with dataframes
        :param populations - List of populationnames, without .csv at the end
        :param path - string path to directory
        """
        self.populations = populations
        self.path = path

    def compare_models_populationwise(self, models: list,  title:str, show:bool=True, dest_path:str=None, showfliers: bool=False):
        """
        BoxPlots of all given models with their macro, micro and weighted f1-scores
        """
        da = Data(self.populations, self.path)
        k_folds = da.k_fold_cross_validation_populationwise()
        k_folds_r = da.k_fold_cross_validation_populationwise(shuffle=True)

        
        data = []
        for m in models:
            model = m[0]
            model_name = m[1]

            for pop in k_folds.keys():
                for k in k_folds[pop].keys():
                    X = k_folds[pop][k]["X_train"] 
                    x = k_folds[pop][k]["X_test"]
                    Y = k_folds[pop][k]["y_train"] 
                    y = k_folds[pop][k]["y_test"]
                    model.set_data(X, x, Y, y)
                    model.train()
                    mi, ma, weigth = model.predict()
                    data.append([model_name, "micro f1-score", mi])
                    data.append([model_name, "macro f1-score", ma])
                    data.append([model_name, "weighted f1-score", weigth])

                    X = k_folds_r[pop][k]["X_train"] 
                    x = k_folds_r[pop][k]["X_test"]
                    Y = k_folds_r[pop][k]["y_train"] 
                    y = k_folds_r[pop][k]["y_test"]
                    model.set_data(X, x, Y, y)
                    model.train()
                    mi, ma, weigth = model.predict()
                    data.append([model_name + "\n shuffled labels", "micro f1-score", mi])
                    data.append([model_name + "\n shuffled labels", "macro f1-score", ma])
                    data.append([model_name + "\n shuffled labels", "weighted f1-score", weigth])

        df = pd.DataFrame(data, columns = ['model', "f1-score", "value"])

        sns.set_theme(palette="pastel")
        sns.boxplot(x="model", y="value",
                    hue="f1-score", 
                    data=df, showfliers=showfliers)
        plt.title(title)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()


svm = SVMclassifier()
ffn = FeedforwardNetWork()

models = [(ffn, "FFN"), (svm, "SVM")]
ok, not_ok = sort_out_populations()
path = r'D:\Dataframes\most_active_neurons\40'
title = "5-Fold Cross Validation results of Support Vector Machine (SVM) and Feedforward Network (FFN): \n population-wise (100 Populations in total) training/testing with 40 most active neurons\n and SMOTE used on training folds"
p = PerformancePlotter(ok, path)
p.compare_models_populationwise(models, title)




