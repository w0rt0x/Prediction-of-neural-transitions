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
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
        iso = Isomap(n_components=dim)
        iso.fit(self.data.T)
        self.reduced_data = iso.transform(self.data.T)

    def do_TSNE(self, dim, seed):
        pass

    def create_binary_transition_labels(self):
        pass

    def create_multiclass_transition_labels(self, remove_last_day=True):
        # remove day 4
        pass

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

pop = "bl693_no_white_Pop05"

a = Preprocessor(pop)
#a.do_ISOMAP(2)
a.get_most_active_neurons()
a.df_to_file(r'C:\Users\Sam\Desktop')