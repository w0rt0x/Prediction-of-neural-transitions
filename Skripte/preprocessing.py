import numpy as np
import pandas as pd
import os
from os.path import isfile, join
from scipy.io import loadmat
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns


class Preprocessor():

    def __init__(self, pop, NaN_replacement=0):
        """
        Takes in Path to directory and population name of file (.csv)
        """
        self.population = pop
        path_class = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_class.mat".format(pop)
        path_lact = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_lact.mat".format(pop)
        mat = loadmat(path_class)
        days = mat['class_all_days'][0]
        trials = mat['class_stim_evo'][0]
        # Creating Header-List with (Day, Trial) Format
        header = list(map(lambda d, t: (d, t), days, trials))

        # Getting Neuron activity
        mat = loadmat(path_lact)
        data = []
        for i in range(len(mat['vecs_all_days'])):
            data.append(mat['vecs_all_days'][i])

        # Swapping x and y coordinats in data so that one trial is in one list (x-axis)
        data = np.array(data).T

        # replacing NaN-values with replacement
        for i in range(len(data)):
            for j in range(len(data[i])):
                if math.isnan(data[i][j]):
                    data[i][j] = NaN_replacement

        self.data, self.header = data.T, header

        # Getting Response Modes
        mat = loadmat(r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\data_for_sam' + '\\' + pop + '.mat')
        array = mat['cl_id']
        self.all_labels = array
        d = dict()
        for i in range(0, 34):
            for j in range(0, 4):
                d[(j+1, i+1)] = array[i][j]
        label = []
        for i in range(len(header)):
            label.append(d[header[i]])

        self.label = np.asarray(label)

    def plot_label(self):
        data = np.zeros((34,4))
        for i in range(len(self.all_labels)):
            for j in range((len(self.all_labels[i]))):
                if self.all_labels[i][j]>0:
                    data[i][j] = 1

        ax = sns.heatmap(data , cmap = 'Greys' , linewidths=1, linecolor='black', xticklabels=['Day ' + str(x) for x in range(1,5)], yticklabels=['stim  ' + str(x) for x in range(1,35)])
        plt.title("Transitions of {}".format(self.population))
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['0', '1'])
        plt.show()

    def do_PCA(self, dim):
        """
        Applyies Scikit PCA on data,
        also saves PCA Loadings as self.loading_matrix
        and the variance as self.PCA_var
        and the PCA components as self.reduced_data
        """
        pca = PCA(n_components=dim)
        pca.fit(self.data)
        # also calculating loadings, Source:
        # https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
        self.loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
        self.PCA_var = sum(pca.explained_variance_ratio_)
        self.reduced_data = pca.components_.T
        
    def do_ISOMAP(self, dim):
        """
        does isomap on given data-set,
        saves manifold data to self.reduced_data
        """
        # Source: https://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/
        # https://towardsdatascience.com/what-is-isomap-6e4c1d706b54
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
        iso = Isomap(n_components=dim)
        iso.fit(self.data.T)
        self.reduced_data = iso.transform(self.data.T)

    def do_TSNE(self, dim, seed=1, perplexity=30):
        # https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a 
        self.reduced_data = TSNE(n_components=dim, random_state=seed, perplexity=perplexity).fit_transform(self.data.T)

    def do_umap(self, dim, seed=1):
        reducer = umap.UMAP(n_components=dim, random_state=seed)
        self.reduced_data = reducer.fit_transform(self.data.T)

    def create_binary_transition_labels(self):
        """
        Creates binary transitions (0 and 1, where 1 is from inactive to active)
        also removes day 4 trials
        """
        transistions = [0] * len(self.label)
        for trail in self.header:
            # Skipping Day 4
            if trail[0] == 4:
                pass
            else:
                next_day = (trail[0] + 1, trail[1])
                # Binary Transition: from 0 to response
                if self.label[self.header.index(next_day)] > 0 and self.label[self.header.index(trail)] == 0:
                    indices = [i for i in range(len(self.header)) if self.header[i] == trail]
                    for j in indices:
                        transistions[j] = 1
        self.label = np.asarray(transistions)

    def create_multiclass_transition_labels(self, skip: int=1):
        """
        Creates multiclass transitions (0->0, 0->1, ...)
        also removes day 4 trials (in case of skip>1, day 3 gets removed too)
        """
        transistions = [0] * len(self.label)
        if skip == 1:
            for trail in set(self.header):
                # Skipping Day 4
                if trail[0] == 4:
                    pass
                else:
                    next_day = (trail[0] + 1, trail[1])
                    indices = [i for i in range(len(self.header)) if self.header[i] == trail]
                    label = '-'
                    if self.label[self.header.index(next_day)] == 0 and self.label[self.header.index(trail)] == 0:
                        label = '0->0'
                    if self.label[self.header.index(next_day)] > 0 and self.label[self.header.index(trail)] > 0:
                        label = '1->1'
                    if self.label[self.header.index(next_day)] > 0 and self.label[self.header.index(trail)] == 0:
                        label = '0->1'
                    if self.label[self.header.index(next_day)] == 0 and self.label[self.header.index(trail)] > 0:
                        label = '1->0'
                    for j in indices:
                        transistions[j] = label
            
        else:
            for trail in set(self.header):
                # Skipping Day 4 and 3
                if trail[0] == 4 or trail[0] == 3:
                    pass
                else:
                    next_day = (trail[0] + 2, trail[1])
                    indices = [i for i in range(len(self.header)) if self.header[i] == trail]
                    label = '-'
                    if self.label[self.header.index(next_day)] == 0 and self.label[self.header.index(trail)] == 0:
                        label = '0->0'
                    if self.label[self.header.index(next_day)] > 0 and self.label[self.header.index(trail)] > 0:
                        label = '1->1'
                    if self.label[self.header.index(next_day)] > 0 and self.label[self.header.index(trail)] == 0:
                        label = '0->1'
                    if self.label[self.header.index(next_day)] == 0 and self.label[self.header.index(trail)] > 0:
                        label = '1->0'
                    for j in indices:
                        transistions[j] = label
            
        self.label = np.asarray(transistions)

    def get_most_active_neurons(self, n=30):
        matrix = self.data
        sums = []
        neurons = []
        for i in range(len(matrix)):
            sums.append(matrix[i].sum())
        # Getting indices:
        # https://www.geeksforgeeks.org/python-indices-of-n-largest-elements-in-list/
        indices = sorted(range(len(sums)), key = lambda sub: sums[sub])[-n:]

        for i in indices:
            neurons.append(matrix[i])

        self.reduced_data = np.array(neurons).T

    def df_to_file(self, path):
        """
        Saves csv pandas dataframe and saves it to given path to directory
        """
        matrix = []
        for i in range(len(self.header)):
            row = [self.header[i]] + self.reduced_data[i].tolist() + [self.label[i].tolist()]
            matrix.append(row)
        cols = ['label']
        for i in range(len(self.reduced_data[0])):
            cols.append('Component ' + str(i + 1))
        cols.append("response")
        dataframe = pd.DataFrame(matrix, columns=cols)
        dataframe.to_csv(path + '\\{}.csv'.format(self.population))

    def minmax_scaler(self):
        """
        Uses minmax-Scaler on data
        """
        self.data = MinMaxScaler().fit_transform(self.data)

    def normalization(self):
        """
        Uses normalisation on data
        """
        self.data = Normalizer().fit_transform(self.data)

    def standard_scaler(self):
        """
        Uses Standard Scaler on data
        """
        self.data = StandardScaler().fit_transform(self.data)

    def plot_data(self, title):
        """
        Does a scatter plot for the 2D reduced Data,
        Source:
        https://pythonspot.com/matplotlib-scatterplot/
        """
        cols = {"0->0" : "cyan", "0->1" : "lime", "1->0" : "red", "1->1" : "yellow"}

        # Create plot
        fig = plt.figure()
        #plt.style.use('dark_background')
        ax = fig.add_subplot(1, 1, 1)

        for i in range(len(self.reduced_data)):
            if self.label[i] != '0':
                x = self.reduced_data[i][0]
                y = self.reduced_data[i][1]
                c = cols[self.label[i]]
                l = self.label[i]
                ax.scatter(x, y, alpha=0.8, c=c,label=l)
        
        plt.ylabel('Component 1')
        plt.xlabel('Component 1')
        plt.title(title)
        plt.legend(cols)
        plt.show()
    
    def get_mean(self):
        """
        Calculates mean of each row
        """
        means = []
        data = self.data.T
        for i in range(len(data)):
            means.append([np.mean(data[i])])

        self.reduced_data = np.asarray(means)

    def get_mean_over_reduced_data(self):
        """
        Calculates mean of each row that has already been reduced (nmost active, PCA, etc)
        """
        means = []
        data = self.reduced_data
        for i in range(len(data)):
            means.append([np.mean(data[i])])

        self.reduced_data = np.asarray(means)

    def get_sum(self):
        """
        Calculates sum of each row
        """
        s = []
        data = self.data.T
        for i in range(len(data)):
            s.append([np.sum(data[i])])

        self.reduced_data = np.asarray(s)

    def get_sum_over_reduced(self):
        """
        Calculates sum of each row that has already been reduced (n-most active, PCA, etc)
        """
        s = []
        data = self.reduced_data
        for i in range(len(data)):
            s.append([np.sum(data[i])])

        self.reduced_data = np.asarray(s)

def prepare_data(destination=r'D:\Dataframes\30_mostActive_Neurons', dim = 20):
    path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])

    # removing dubs
    populations = list(populations)
    for pop in populations:
        a = Preprocessor(pop)
        a.get_most_active_neurons(n=40)
        a.create_multiclass_transition_labels(skip=2)
        a.df_to_file(destination)
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))


prepare_data(destination=r'D:\Dataframes\double_skip', dim=40)



