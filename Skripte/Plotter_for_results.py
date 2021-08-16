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

    def compare_models_populationwise(self, models: list,  title:str, preprocess: bool=False, show:bool=True, dest_path:str=None, showfliers: bool=False):
        """
        BoxPlots of all given models with their macro, micro and weighted f1-scores
        :param models (list) - list of models in form of [(model, "Title of model"), ...]
        :param title (str) - title of plot
        :param preprocess (bool, default False) - if True, uses standard-scaling on data before training model, recommended for tSNE data
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        :param show_outliers (bool, default is False) - if True: shows outliers of Boxplot (1.5 times outside Inter-Quartile range) 
        """
        da = Data(self.populations, self.path)
        k_folds = da.k_fold_cross_validation_populationwise()
        k_folds_r = da.k_fold_cross_validation_populationwise(shuffle=True)

        
        data = []
        for m in models:
            model = m[0]
            model_name = m[1]

            for pop in k_folds.keys():
                print(pop)
                for k in k_folds[pop].keys():
                    X = k_folds[pop][k]["X_train"] 
                    x = k_folds[pop][k]["X_test"]
                    Y = k_folds[pop][k]["y_train"] 
                    y = k_folds[pop][k]["y_test"]
                    model.set_data(X, x, Y, y)
                    if preprocess:
                        model.preprocess()
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
                    if preprocess:
                        model.preprocess()
                    model.train()
                    mi, ma, weigth = model.predict()
                    data.append([model_name + "\nshuffled labels", "micro f1-score", mi])
                    data.append([model_name + "\nshuffled labels", "macro f1-score", ma])
                    data.append([model_name + "\nshuffled labels", "weighted f1-score", weigth])

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

    def CM_for_all_pop(self, model, title:str, norm: bool=True, show:bool=True, dest_path:str=None, preprocess: bool=False):
        """
        :param model - FFN or SVM
        :param Title (str) title
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        :param preprocess (bool, default False) - if True, uses standard-scaling on data before training model, recommended for tSNE data
        """
        CM = np.zeros((4, 4))
        for pop in self.populations:
            d = Data([pop], self.path)
            d.split_trial_wise()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            model.set_data(X, x, Y, y)
            if preprocess:
                model.preprocess()
            model.train()
            model.predict()
            CM = CM + model.get_CM()

        if norm:
            CM_norm = np.zeros((4, 4))
            for row in range(len(CM)):
                for col in range(len(CM[row])):
                    CM_norm[row][col] = round(CM[row][col] / np.sum(CM[row]), 3)
            CM = CM_norm


        df_cm = pd.DataFrame(CM_norm, index = [i for i in ['0->0', '0->1', '1->0', '1->1']],
                  columns = [i for i in ['0->0', '0->1', '1->0', '1->1']])
        #plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path)

        plt.clf()
        plt.cla()
        plt.close()

    def compare_transitions(self, model, title:str, day2_transitions: str=r'D:\Dataframes\double_skip', show:bool=True, dest_path:str=None, showfliers: bool=False):
        """
        Compares prediction quality of a model regarding the transitions between 1 day and 2 days
        :param models (list) - list of models in form of [(model, "Title of model"), ...]
        :param title (str) - title of plot
        :param second_path (str) - different directory with mean or summed neural data
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        :param showfliers (bool, default is False) - if True: shows outliers of Boxplot (1.5 times outside Inter-Quartile range) 
        """
        data = []
        for pop in self.populations:
            d = Data([pop], self.path)
            d.split_trial_wise()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            model.set_data(X, x, Y, y)
            model.train()
            mi, ma, weigth = model.predict()
            data.append(["Transition\nover 1 day", "micro f1-score", mi])
            data.append(["Transition\nover 1 day", "macro f1-score", ma])
            data.append(["Transition\nover 1 day", "weighted f1-score", weigth])

            d = Data([pop], day2_transitions)
            d.split_trial_wise()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            model.set_data(X, x, Y, y)
            model.train()
            mi, ma, weigth = model.predict()
            data.append(["Transition\nover 2 days", "micro f1-score", mi])
            data.append(["Transition\nover 2 days", "macro f1-score", ma])
            data.append(["Transition\nover 2 days", "weighted f1-score", weigth])

        df = pd.DataFrame(data, columns = ['Transition-type', "f1-score", "value"])

        sns.set_theme(palette="pastel")
        sns.boxplot(x="f1-score", y="value",
                    hue="Transition-type", 
                    data=df, showfliers=showfliers)
        plt.title(title)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def compare_models_across_populations(self, models: list,  title:str, show:bool=True, dest_path:str=None, showfliers: bool=False):
        """
        BoxPlots of all given models with their macro, micro and weighted f1-scores
        but shared across populations
        :param models (list) - list of models in form of [(model, "Title of model"), ...]
        :param title (str) - title of plot
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        :param showfliers (bool, default is False) - if True: shows outliers of Boxplot (1.5 times outside Inter-Quartile range) 
        """
        da = Data(self.populations, self.path)
        k_folds = da.k_fold_cross_validation()
        k_folds_r = da.k_fold_cross_validation(shuffle=True)

        
        data = []
        for m in models:
            model = m[0]
            model_name = m[1]

            for k in k_folds.keys():
                X = k_folds[k]["X_train"] 
                x = k_folds[k]["X_test"]
                Y = k_folds[k]["y_train"] 
                y = k_folds[k]["y_test"]
                model.set_data(X, x, Y, y)
                model.train()
                mi, ma, weigth = model.predict()
                data.append([model_name, "micro f1-score", mi])
                data.append([model_name, "macro f1-score", ma])
                data.append([model_name, "weighted f1-score", weigth])

                X = k_folds_r[k]["X_train"] 
                x = k_folds_r[k]["X_test"]
                Y = k_folds_r[k]["y_train"] 
                y = k_folds_r[k]["y_test"]
                model.set_data(X, x, Y, y)
                model.train()
                mi, ma, weigth = model.predict()
                data.append([model_name + "\nshuffled labels", "micro f1-score", mi])
                data.append([model_name + "\nshuffled labels", "macro f1-score", ma])
                data.append([model_name + "\nshuffled labels", "weighted f1-score", weigth])

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


ffn = FeedforwardNetWork()
#models = [(ffn, "FFN")]
svm = SVMclassifier()
ok, not_ok = sort_out_populations()
path = r'D:\Dataframes\most_active_neurons\40'
title = "Class-wise Normalized Confusion Matrix of all Populations with 4 classes.\n Classification via SVM(rbf-Kernel, c=1, gamma=0.5, balanced class-weights)\n SMOTE used on training-data"
title = "Class-wise Normalized Confusion Matrix of all Populations with 4 classes.\n Classification via FFN, SMOTE used on training-data"
p = PerformancePlotter(ok, path)
p.CM_for_all_pop(ffn, title)
#p.compare_models_across_populations(models, title)
#p.compare_transitions(svm, title)
#p.compare_models_populationwise(models, title, preprocess=True)






