from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from Classifier import SVMclassifier
from collections import Counter
from copy import deepcopy
from data_holder import Data
import warnings
warnings.filterwarnings('always')
from getter_for_populations import get_all_pop, sort_out_populations


class Plotter:
    """
    Plotter for general visualisation
    """
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
            
            yellow = mpatches.Patch(color='magenta', label='1->1')
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
            
            yellow = mpatches.Patch(color='magenta', label='1->1')
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

    def compare_n_neurons(self, title: str, model: SVMclassifier, neurons:list=list(range(5, 101, 5))):
        """
        Compares performance of n most active neurons
        :param title (str) - Title of plot
        :param model (SVM) - Model that gets tested
        :param neurons (list) - List of integers that are directory names of path in init function
        """
        macro = []
        micro = []
        weighted = []
        macro_r = []
        micro_r = []
        weighted_r = []

        for n in neurons:

            # normal labels
            d = Data(self.populations, self.path + str(n))
            d.split_trial_wise()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            model.set_data(X, x, Y, y)
            model.train()
            mi, ma, weigth = model.predict()
            macro.append(ma)
            micro.append(mi)
            weighted.append(weigth)
            
            # Randomized labels
            d = Data(self.populations, self.path + str(n))
            d.split_trial_wise()
            d.shuffle_labels()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            model.set_data(X, x, Y, y)
            model.train()
            mi, ma, weigth = model.predict()
            macro_r.append(ma)
            micro_r.append(mi)
            weighted_r.append(weigth)

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

    def plot_actual_vs_predicted(self, method: str, x_axis: str, y_axis:str, preprocess: bool=False, show: bool=True, dest_path: str=None):
        """
        Plots 2 Plots: Predicted(SVM) vs Actual data
        :param model (SVM, ffn) - Models that gets tested in format [(model, "name of model"), ...]
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param preprocess (bool) - preprocesses data before using SVM (only recommended for tSNE)
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
        """
        
        for pop in self.populations:
            # Getting Data
            d = Data([pop], self.path)
            d.split_trial_wise()
            d.use_SMOTE()
            X, x, Y, y = d.get_data()
            x_t = x.T
            # Actual Data

            figure, axis = plt.subplots(2, 2, figsize=(13,7))
            y_col = self.__multiclass_to_color(y.tolist())
            axis[0][0].scatter(x_t[0], x_t[1], c=y_col, alpha=0.5)
            axis[0][0].set_title("Actual Data")
            axis[0][0].set_xlabel(x_axis)
            axis[0][0].set_ylabel(y_axis)
            
            svm = SVMclassifier()
            svm.set_data(X, x, Y, y)
            if preprocess:
                svm.preprocess()
            svm.train()
            svm.predict()
            pred = self.__multiclass_to_color(svm.get_predictions().tolist())
            axis[0][1].scatter(x_t[0], x_t[1], c=pred, alpha=0.5)
            axis[0][1].set_title("{} Prediction".format(svm.get_info()))
            axis[0][1].set_xlabel(x_axis)
            axis[0][1].set_ylabel(y_axis)

            svm = SVMclassifier(kernel="linear")
            svm.set_data(X, x, Y, y)
            if preprocess:
                svm.preprocess()
            svm.train()
            svm.predict()
            pred = self.__multiclass_to_color(svm.get_predictions().tolist())
            axis[1][0].scatter(x_t[0], x_t[1], c=pred, alpha=0.5)
            axis[1][0].set_title("{} Prediction".format(svm.get_info()))
            axis[1][0].set_xlabel(x_axis)
            axis[1][0].set_ylabel(y_axis)

            svm = SVMclassifier(kernel="poly")
            svm.set_data(X, x, Y, y)
            if preprocess:
                svm.preprocess()
            svm.train()
            svm.predict()
            pred = self.__multiclass_to_color(svm.get_predictions().tolist())
            axis[1][1].scatter(x_t[0], x_t[1], c=pred, alpha=0.3)
            axis[1][1].set_title("{} Prediction".format(svm.get_info()))
            axis[1][1].set_xlabel(x_axis)
            axis[1][1].set_ylabel(y_axis)

            yellow = mpatches.Patch(color='magenta', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            figure.legend(handles=[yellow, red, green, cyan], bbox_to_anchor=(0.49, 0.90))

            figure.suptitle("Actual and Predicted Data ({}) of {}".format(method, pop))
            figure.tight_layout()

            if show:
                plt.show() 
            if dest_path !=None:
                plt.savefig(dest_path + '\\{}.png'.format(pop))

            plt.clf()
            plt.cla()
            plt.close()

    def boxplots_of_classes(self, title:str, y_axis: str="mean activity over all neurons", second_path: str=r'D:\Dataframes\double_skip_mean', show:bool=True, dest_path:str=None, show_outliers: bool=False):
        """
        makes 4 Box-plots that show the first component of the dataframe (mean, median, etc) of the 4 classes
        """
        data = []
        counter = 0
        for pop in self.populations:
            df = pd.read_csv(self.path + '\\{}.csv'.format(pop))
            trials = df['label'].tolist()
            values = df['Component 1'].tolist()
            response = df['response'].tolist()
            
            for i in range(len(response)):
                # Removing day 4 trials
                if eval(trials[i])[0] != 4:
                    data.append([response[i], values[i], "Transition over 1 day"])

            df = pd.read_csv(second_path + '\\{}.csv'.format(pop))
            trials = df['label'].tolist()
            values = df['Component 1'].tolist()
            response = df['response'].tolist()
            
            for i in range(len(response)):
                # Removing day 3 and 4 trials
                if eval(trials[i])[0] != 4 and eval(trials[i])[0] != 3:
                    data.append([response[i], values[i], "Transition over 2 days"])

        df = pd.DataFrame(data, columns = ['Labels', y_axis, "Transition"])

        self.__box_plot(df, "Labels", y_axis, "Transition", title, show=show, dest_path=dest_path, showfliers=show_outliers, order = ["0->0", "0->1", "1->0", "1->1"])

    def __box_plot(self, df: pd.DataFrame, x:str, y:str, hue:str, title:str, show: bool=True, dest_path: str=None, showfliers = False, order: list=None):

        sns.set_theme(palette="pastel")
        sns.boxplot(x=x, y=y,
                    hue=hue, order=order, 
                    data=df, showfliers=showfliers)
        plt.title(title)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def histogram_single_values(self, x_axis:str, title:str, max_bins:int=.1, density: bool=False, show:bool=True, dest_path:str=None):
        """
        Plots the Distribution of means or sums (users choice) as a histogram
        Compares multiple models
        :param x_axis (str) Label
        :param Title (str) title
        :param max_bins(bool) - is max number of bins
        :param density (bool, default is False) - plots seaborn density plot
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
            plt.hist(d[labels[i]], bins, alpha=0.5, label=labels[i], color=colors[i], histtype='step', density=density)
            plt.axvline(np.array(list(d[labels[i]])).mean(), ls='--', color=colors[i], linewidth=1, label="{} mean".format(labels[i]))
        plt.title(title)
        plt.xlabel(x_axis)
        if density:
            plt.ylabel("density")
        else:
            plt.ylabel("Occurences")
        plt.legend(loc='upper right')

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()

    def plot_mean_of_each_neuron(self, title:str, show:bool=True, dest_path:str=None, std=True):
        """
        Line Plot that shows neuron-wise mean, with or without standard deviation
        """
        d = Data(self.populations, self.path)
        d.split_trial_wise()
        X, x, Y, y = d.get_data()
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))

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

#ok, not_ok  = sort_out_populations()
#p = Plotter(ok, r'D:\Dataframes\tSNE\perp30')
#dest = r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Bachelor-ML\Skripte\Plots\Prediction results, Grid Searches and parameter estimation\Prediction of next Day\Actual data vs predicted\tSNE(preprocessed)'
#p.plot_actual_vs_predicted("2 tSNE Components (preprocessed)","first tSNE Component", "second tSNE Component", show=False, dest_path=dest, preprocess=True)
#p.plot_mean_of_each_neuron("Neuron-wise mean and standard-deviation of the 40 Most active neurons,\n seperated into the four classes")