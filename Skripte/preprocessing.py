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
        d = dict()
        for i in range(0, 34):
            for j in range(0, 4):
                d[(j+1, i+1)] = array[i][j]
        label = []
        for i in range(len(header)):
            label.append(d[header[i]])

        self.label = np.asarray(label)

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

        # MEHR PARAMETER
        """
        # Source: https://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/
        # https://towardsdatascience.com/what-is-isomap-6e4c1d706b54
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
        iso = Isomap(n_components=dim)
        iso.fit(self.data.T)
        self.reduced_data = iso.transform(self.data.T)

    def do_TSNE(self, dim, seed=1):
        # https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a 
        self.reduced_data = TSNE(n_components=dim, random_state=seed).fit_transform(self.data.T)

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

    def create_multiclass_transition_labels(self, remove_last_day=True):
        """
        Creates multiclass transitions (0->0, 0->1, ...)
        also removes day 4 trials
        """
        transistions = [0] * len(self.label)
        for trail in self.header:
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
        a.do_TSNE(dim)
        a.create_multiclass_transition_labels()
        a.df_to_file(destination)
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))

prepare_data(destination=r'D:\Dataframes\tSNE\multi_2d', dim=2)

#pop = "bl693_no_white_Pop05"
#a = Preprocessor(pop)
#a.do_TSNE(2)
#a.create_multiclass_transition_labels()
##a.get_most_active_neurons()
#a.df_to_file(r'C:\Users\Sam\Desktop')
