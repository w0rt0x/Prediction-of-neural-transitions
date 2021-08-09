from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import os
from os.path import isfile, join
from Classifier import Classifier
from collections import Counter
from copy import deepcopy
import warnings
warnings.filterwarnings('always')



class Plotter:

    def __init__(self, populations: list, path:str):
        """
        Needs List of populations and directory with dataframes
        :param populations - List of populationnames, without .csv at the end
        :param path - string path to directory
        """
        self.populations = populations
        self.path = path
        full_paths = []
        for i in range(len(populations)):
            full_paths.append(path + '\\' + populations[i] + '.csv')
        self.full_paths = full_paths

    def plot_2D(self, method: str, x_axis: str, y_axis: str, show: bool=True, dest_path: str=None):
        """
        Plots 2D Data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
        """
        for pop in self.full_paths:
            df = pd.read_csv(pop)
            # Getting Data
            header = df['label'].tolist()
            x = df['Component 1'].tolist()
            y = df['Component 2'].tolist()
            label = df['response'].tolist()
            #Removing Day 4
            i = header.index('(4, 1)')
            x = x[:i]
            y = y[:i]
            label = label[:i]
            for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'magenta'

            # Plotting
            plt.scatter(x, y, c=label, alpha=0.3)
            name = self.populations[self.full_paths.index(pop)]
            plt.title('{} of {}'.format(method, name))
            
            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            plt.legend(handles=[yellow, red, green, cyan])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            if show:
                plt.show()
            if dest_path != None:
                plt.savefig(dest_path + '\\{}.png'.format(name))
            
            plt.clf()
            plt.cla()
            plt.close()

    def plot_3D(self, method: str, x_axis: str, y_axis: str, z_axis: str, show: bool=True, dest_path: str=None):
        """
        Plots 2D Data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param z_axis (str) - name of z-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
        """
        for pop in self.full_paths:
            fig = plt.figure()
            ax = plt.axes(projection ='3d')
            df = pd.read_csv(pop)
            # Getting Data
            header = df['label'].tolist()
            x = df['Component 1'].tolist()
            y = df['Component 2'].tolist()
            z = df['Component 3'].tolist()
            label = df['response'].tolist()
            #Removing Day 4
            i = header.index('(4, 1)')
            x = x[:i]
            y = y[:i]
            z = z[:i]
            label = label[:i]
            for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'magenta'

            # Plotting
            ax = plt.axes(projection ="3d")
            ax.scatter(x, y, z, c=label, alpha=0.5)
            name = self.populations[self.full_paths.index(pop)]
            ax.set_title('{} of {}'.format(method, name))
            
            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            plt.legend(handles=[yellow, red, green, cyan])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_zlabel(z_axis)
            if show:
                plt.show()
            if dest_path != None:
                plt.savefig(dest_path + '\\{}.png'.format(name))

            plt.clf()
            plt.cla()
            plt.close()

    def compare_n_neurons(self, title: str, neurons:list=list(range(5, 101, 5)), kernel:str='rbf', degree:int=3, c:float=1.0, gamma:float=1.0):
        """
        Compares performance of n most active neurons
        :param title (str) - Title of plot
        :param neurons (list) - List of integers that are directory names of path in init function
        """
        macro = []
        micro = []
        weighted = []
        macro_r = []
        micro_r = []
        weighted_r = []

        for n in neurons:
            # Regular run
            a = Classifier(self.populations, self.path + str(n))
            a.split_trial_wise()
            a.use_SMOTE()
            a.do_SVM(kernel=kernel, c=c, gamma=gamma, degree=degree, class_weight='balanced')
            report = a.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])
            
            # Randomized labels
            r = Classifier(self.populations, self.path + str(n))
            r.split_trial_wise()
            r.use_SMOTE()
            r.shuffle_labels()
            r.do_SVM(kernel=kernel, c=c, gamma=gamma, degree=degree, class_weight='balanced')
            report = r.get_report()
            macro_r.append(report['macro avg']['f1-score'])
            micro_r.append(report['accuracy'])
            weighted_r.append(report['weighted avg']['f1-score'])

        
        plt.plot(neurons,macro, marker = 'o', color='#f70d1a', label="Macro F1")
        plt.plot(neurons,micro, marker = 'x', color='#08088A', label="Micro F1")
        plt.plot(neurons,weighted, marker = '+', color='#FFBF00', label="weighted F1")
        plt.plot(neurons,macro_r, marker = 'o', color='#f70d1a', label="Macro F1 (random)", linestyle = '--')
        plt.plot(neurons,micro_r, marker = 'x', color='#08088A', label="Micro F1 (random)", linestyle = '--')
        plt.plot(neurons,weighted_r, marker = '+', color='#FFBF00', label="weighted F1 (random)", linestyle = '--')
        plt.xlabel("#Neurons")
        plt.xticks(neurons)
        plt.ylabel("F1-Scores")
        plt.ylim([0, 1])
        plt.legend(loc="upper left")
        plt.title(title)
        plt.show()
        
    def plot_actual_vs_predicted(self, method: str, x_axis: str, y_axis: str, show: bool=True, dest_path: str=None):
        """
        Plots 2 Plots: Predicted(SVM) vs Actual data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
        """
        for pop in self.populations:
            c = Classifier([pop], self.path)
            c.split_trial_wise()
            c.use_SMOTE()
            c.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            X, x, Y, y = c.get_data()
            x = x.T

            y = self.__multiclass_to_color(y.tolist())
            pred = self.__multiclass_to_color(c.get_predictions().tolist())

                # Plotting the Data
            figure, axis = plt.subplots(1, 2, figsize=(13,7))
            axis[0].scatter(x[0], x[1], c=y, alpha=0.5)
            axis[0].set_title("Actual Data")
            axis[0].set_xlabel(x_axis)
            axis[0].set_ylabel(y_axis)
                
            # For Cosine Function
            axis[1].scatter(x[0], x[1], c=pred, alpha=0.5)
            axis[1].set_title("Prediction")
            axis[1].set_xlabel(x_axis)
            axis[1].set_ylabel(y_axis)

            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            figure.legend(handles=[yellow, red, green, cyan])

            figure.suptitle("{} - Test-Data of {},\n SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')\n\n".format(method, pop))
            

            if show:
                plt.show() 
            if dest_path !=None:
                plt.savefig(dest_path + '\\{}.png'.format(self.populations[0]))

            plt.clf()
            plt.cla()
            plt.close()

    def __multiclass_to_color(self, label):
        for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'magenta'
        return label

    def compare_models(self, pops: list, paths:list, x_axis:str, title:str, width: float=0.2, show:bool=True, dest_path:str=None, names = None):
        """
        Compares multiple models
        :param pops (list of lists with str (population names))
        :param paths (list of paths)
        :param x_axis (str) Label
        :param Title (str) title
        """
        micro = []
        macro = []
        weighted = []
        pops.append(self.populations)
        paths.append(self.path)
        if names == None:
            names = []
            for i in range(len(pops)):
                names.append('\n'.join(pops[i]))
                names.append('\n'.join(pops[i]) + '\n(random labels)')
        
        for i in range(len(pops)):
            # möglicher Fehler?
            c = Classifier([pops[i]], paths[i])
            c.split_trial_wise()
            c.use_SMOTE()
            c.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = c.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])
                
            # Randomized labels
            r = Classifier(self.populations, self.path)
            r.split_trial_wise()
            r.use_SMOTE()
            r.shuffle_labels()
            r.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = r.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])

        x = np.arange(0, len(names))
        plt.bar(x, micro, width=width, color='yellow', label="Micro F1-Score")
        plt.bar(x + width, macro, width=width, color='blue', label="Macro F1-Score")
        plt.bar(x + 2*width, weighted, width=width, color='green', label="Weighted F1-Score")
        plt.xticks(x, names)
        plt.ylabel("Scores")
        plt.xlabel(x_axis)
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.legend()
        plt.title(title)
        if show:
            plt.show()
        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def boxplots_of_classes(self, y_axis:str, title:str, show:bool=True, dest_path:str=None, show_outliers: bool=False):
        """
        makes 4 Box-plots that show the first component of the dataframe (mean, median, etc) of the 4 classes
        """
        d = {}
        for pop in self.populations:
            df = pd.read_csv(self.path + '\\{}.csv'.format(pop))
            trials = df['label'].tolist()
            values = df['Component 1'].tolist()
            response = df['response'].tolist()
            
            for i in range(len(response)):
                if response[i] in d:
                    d[response[i]].add(values[i])
                else:
                    # Removing day 4 trials
                    if eval(trials[i])[0] != 4:
                        d[response[i]] = set()
                        d[response[i]].add(values[i])

        data = []
        
        labels = ['0->0', '0->1', '1->1', '1->0']
        for key in labels:
            data.append(list(d[key]))
        
        plt.boxplot(data, labels=labels, showfliers=show_outliers) 
        plt.xlabel('class')
        plt.ylabel(y_axis)
        plt.title(title)
        plt.grid(axis='y')

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def sort_out_populations(self, percent:float=0.0, num_class:int=4, show:bool=True, dest_path:str=None):
        """
        Checks and plots the number of Populations hat have n classes and a specific percentage of each class
        """
        ok = []
        not_ok = []

        for pop in self.populations:
            df = pd.read_csv(self.path + '\\{}.csv'.format(pop))
            liste = df['response'].tolist()

            response = list(set(liste))
            response.remove('0')

            if len(response) != num_class:
                not_ok.append(pop)
            else:
                if percent == 0.0:
                    ok.append(pop)
                else:
                    occurences = Counter(liste)
                    # remove day 4
                    del occurences["0"]
                    keys = occurences.keys()
                    s = 0
                    for key in keys:
                        s = s + occurences[key]
                    for key in keys:
                        occurences[key] = occurences[key] / s
                    
                    add = True
                    for key in keys:
                        if occurences[key] < percent:
                            add = False
                    if add:
                         ok.append(pop)
                    else:
                        not_ok.append(pop)
            

        classes = ["All {} classes present\n ({} in total)".format(num_class, len(ok)), "not all classes present\n ({} in total)".format(len(not_ok))]
        plt.pie([len(ok),(len(not_ok))], startangle=90, colors=['#5DADE2', '#515A5A'], labels=classes, autopct='%.1f%%')
        if percent == 0.0:
            plt.title("Valid Populations that have all {} classes".format(num_class))
        else:
            plt.title("Valid Populations that have all {} classes and \n each class has #class_i/#total >= {}".format(num_class, percent))
        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format("Valid Populations"))

        plt.clf()
        plt.cla()
        plt.close()

        return ok, not_ok

    def histogram_single_values(self, x_axis:str, title:str, max_bins:int=1, show:bool=True, dest_path:str=None):
        """
        Plots the Distribution of means or sums (users choice) as a histogram
        Compares multiple models
        :param x_axis (str) Label
        :param Title (str) title
        :param max_bins(bool) - is max number of bins
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        """
        d = {}
        for pop in self.populations:
            df = pd.read_csv(self.path + '\\{}.csv'.format(pop))
            trials = df['label'].tolist()
            values = df['Component 1'].tolist()
            response = df['response'].tolist()
            
            for i in range(len(response)):
                if response[i] in d:
                    d[response[i]].add(values[i])
                else:
                    # Removing day 4 trials
                    if eval(trials[i])[0] != 4:
                        d[response[i]] = set()
                        d[response[i]].add(values[i])

        data = []
        labels = d.keys()
        if len(labels)==4:
            labels = ['0->0', '0->1', '1->1', '1->0']
        for key in labels:
            data.append(list(d[key]))
        
        bins = np.linspace(0, max_bins, 500)
        colors = self.__multiclass_to_color(deepcopy(labels))
        for i in range(len(labels)):
            plt.hist(d[labels[i]], bins, alpha=0.5, label=labels[i], color=colors[i], histtype='step', density=True)
            plt.axvline(np.array(list(d[labels[i]])).mean(), ls='--', color=colors[i], linewidth=1, label="{} mean".format(labels[i]))
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel("Occurences")
        plt.legend(loc='upper right')

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def __get_f1s(self, K, random:bool=False) -> Tuple[list, list, list]:
        micro = []
        macro = []
        weighted = []
        a = Classifier(self.populations, self.path)
        data = a.k_fold_cross_validation_populationwise(K=K, kernel=self.kernel, c=self.c, gamma=self.gamma, shuffle=random)
        for pop in data:
            micro.append(data[pop]['MeanMicro'])
            macro.append(data[pop]['MeanMacro'])
            weighted.append(data[pop]['MeanWeighted'])

        return micro, macro, weighted

    def histogram_of_scores(self, title:str, random:bool=False, show:bool=True, dest_path:str=None):
        """
        Plots weighted, macro and micro f1 Scores of provided populations as a Histogram
        :param Title (str) title
        :param random (bool) - default is false, if true: labels get shuffled
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        """
        micro, macro, weighted = self.__get_f1s(self.populations, self.path, random=random)
        
        bins = np.linspace(0, 1, 200)

        plt.hist(micro, bins, alpha=0.5, label="Micro F1-Score", color="red")
        plt.hist(macro, bins, alpha=0.5, label="Macro F1-Score", color="blue")
        plt.hist(weighted, bins, alpha=0.5, label="weighted F1-Score", color="gold")
        plt.title(title)
        plt.xlim(0, 1)
        plt.grid()
        plt.xlabel("Scores")
        plt.ylabel("Occurences")
        plt.legend(loc='upper right')
        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def boxplot_of_scores(self, title:str, show:bool=True, dest_path:str=None, K: int=5):
        """
        Shows f1 scores (with and without shuffled labels) as boxplots
        :param Title (str) title
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        :param K (int) - K for k-Fold Cross Validation
        """
        labels = ["Micro F1", "Micro F1\n shuffled labels", "Macro F1", "Macro F1\n shuffled labels", "Weighted F1", "Weighted F1\n shuffled labels"]
        micro, macro, weighted = self.__get_f1s(K)
        micro_r, macro_r, weighted_r = self.__get_f1s(K, random=True)
        data = [micro, micro_r, macro, macro_r, weighted, weighted_r]
        plt.boxplot(data, labels=labels) 
        plt.xlabel('Type of F1-Score')
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(axis='y')
        plt.ylim(0, 1)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def __get_cm(self, pop:str, path:str) -> np.array:
        """
        returns CM of classification
        """
        c = Classifier([pop], path)
        c.split_trial_wise()
        #c.use_SMOTE()
        c.do_SVM(kernel=self.kernel, c=self.c, gamma=self.gamma, degree=self.degree, class_weight='balanced')
        return c.get_cm()

    def set_svm_parameter(self, kernel: str='rbf', c:float=1.0, gamma:float=1.0, degree:int=3):
        """
        sets SVM Parameters
        """
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.degree = degree

    def CM_for_all_pop(self, title:str, norm: bool=True, show:bool=True, dest_path:str=None):
        """
        :param Title (str) title
        :param show (bool) - Optional, shows plot of true (default is true)
        :param dest_path (str) - saves plot to that directory if provided
        """
        CM = np.zeros((4, 4))
        for pop in self.populations:
            print(self.populations.index(pop), pop)
            CM = CM + self.__get_cm(pop, self.path)

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

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path)

        plt.clf()
        plt.cla()
        plt.close()

    def plot_mean_of_each_class(self, title:str, show:bool=True, dest_path:str=None, std=True):
        """
        Line Plot that shows neuron-wise mean, with or without standard deviation
        """
        a = Classifier(self.populations, self.path)
        a.split_trial_wise()
        X = np.concatenate((a.X_train, a.X_test))
        Y = np.concatenate((a.y_train, a.y_test))

        d = {}
        for i in range(len(Y)):
            if Y[i] in d:
                d[Y[i]].append(X[i])
            else:
                d[Y[i]] = [X[i]]

        stds = {}
        for key in d.keys():
            d[key] = np.asarray(d[key], dtype=float)
            stds[key] = np.std(d[key], axis=0)[::-1]
            d[key] = np.mean(d[key], axis=0)[::-1]
            

        c = {"1->1": "magenta", "0->0": "cyan", "1->0":"red", "0->1": "green"}
        fig, ax = plt.subplots()
        for key in d.keys():
            ax.plot(range(1, len(X[0]) + 1), d[key], color=c[key], label="mean of {}".format(key))
            if std:
                ax.fill_between(range(1, len(X[0]) + 1), d[key] + stds[key], alpha=0.1, color=c[key], label="std of {}".format(key))
        
        plt.xlabel('{} most active Neurons'.format(len(X[0])))
        plt.ylabel("Neuron-wise mean per class")
        plt.title(title)
        plt.legend()

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

def get_all_pop(path: str=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'):
    """
    returns all population-names
    """
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])
    return list(populations)

populations = get_all_pop()
a = Plotter(populations, r'D:\Dataframes\PCA\2')
ok, nt_ok = a.sort_out_populations(show=False)
b = Plotter(ok, r'D:\Dataframes\most_active_neurons\40')
#b = Plotter(ok, r'D:\Dataframes\most_active_neurons\100')
#b.set_svm_parameter(gamma=0.5)
b.boxplot_of_scores("Mean F1-Scores of 5-fold Cross-Validation using the 40 most active neurons \n Classification via Feedforward Network, SMOTE used on Training-Data")
#b.plot_mean_of_each_class("Neuron-wise mean and standard-deviation of the 40 Most active neurons,\n seperated into the four classes")
#b.histogram_single_values("All trials with their mean over all neurons", "Histogram of all populations with all four classes", max_bins=0.1)

#b.boxplot_of_scores("F1-Scores with 40 most active neurons\n and SVM('rbf'-Kernel, balanced class weights) and SMOTE on Training-Data")
#b.histogram_of_scores("Distribution of F1-Scores with 40 most active neurons\n and SVM('rbf'-Kernel, balanced class weights) and SMOTE on Training-Data", random=True)
#b.histogram_single_values("sum over all neurons", "Histogram of all Populations with all 4 Classes \n and a relative frequency of at least 0.05", max_bins=20)
#b = Plotter(ok, r'D:\Dataframes\single_values\mean_over_all')
#b.boxplots_of_classes("Mean activity over all neurons", "All 100 Populations with 4 Classes")
#a.compare_models([['bl693_no_white_Pop02', 'bl693_no_white_Pop03', 'bl693_no_white_Pop05']], [r'D:\Dataframes\tSNE\3D_perp30'], "Input Populations", "Prediction of Multiclass Labels with \n SVM(rbf Kernel, balanced class weights)\n and t-SNE 3D Data")
#a.plot_2D("t-SNE", "t-SNE Component 1", "t-SNE Component 2")
#a.plot_actual_vs_predicted("t-SNE", "Component 1", "Component 2")
#b = Plotter(get_all_pop(), r'D:\Dataframes\tSNE\perp30')
#b.plot_actual_vs_predicted("t-SNE", "t-SNE Component 1", "t-SNE Component 2", show=False, dest_path=r'D:\Dataframes\tSNE\2D_actual_vs_predicted')
