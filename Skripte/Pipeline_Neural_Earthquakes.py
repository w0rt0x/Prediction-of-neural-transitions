import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
from os.path import isfile, join
from scipy.io import loadmat
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px


class NeuralEarthquake_singlePopulation():

    def __init__(self, population, reduction_method, dimension=20, path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', replacement=0):
        self.dimension = dimension
        self.population = population
        self.reduction_method = reduction_method
        self.path_class = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_class.mat".format(
            self.population)
        self.path_lact = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_lact.mat".format(
            self.population)
        self.dictionary = None
        self.data = None
        self.header = None
        self.trials = None
        self.dataframe = None
        self.replacement = replacement  # replacement for NaN

    def set_dim(self, new):
        self.dimension = new

    def set_replacement(self, new):
        self.replacement = new

    def set_reduction_method(self, new):
        self.reduction_method = new

    def read_population(self):
        """
        reads in matlab file, returns lists
        replaces NaN values
        """
        mat = loadmat(self.path_class)
        days = mat['class_all_days'][0]
        trials = mat['class_stim_evo'][0]
        # Creating Header-List with (Day, Trial) Format
        header = list(map(lambda d, t: (d, t), days, trials))

        # Getting Neuron activity
        mat = loadmat(self.path_lact)
        data = []
        for i in range(len(mat['vecs_all_days'])):
            data.append(mat['vecs_all_days'][i])

        # Swapping x and y coordinats in data so that one trial is in one list (x-axis)
        header, data = np.array(header), np.array(data).T

        # replacing NaN-values with replacement
        for i in range(len(data)):
            for j in range(len(data[i])):
                if math.isnan(data[i][j]):
                    data[i][j] = self.replacement

        self.data, self.header = data.T, header

    def data_to_dictionary(self):
        """
        Extracts data from matlab files, saves data as a dictionary-class object
        with (day, stimulus) : [Trails  x Neurons]
        """
        # Saving data to dictionary
        data = self.data.T
        d = dict()

        for i in range(len(self.header)):
            if tuple(self.header[i]) in d:
                # Tuple conversion because numpy-arrays can't be hashed in a dictionary
                d[tuple(self.header[i])] = d[tuple(self.header[i])] + [data[i]]

            else:
                d[tuple(self.header[i])] = [data[i]]

        # Transpose Matrix, so that Neurons x Trials
        trials = list(d.keys())
        for k in trials:
            d[k] = np.array(d[k]).T

        self.dictionary = d
        del data

    def get_single_stim(self, day, stimulus):
        """
        returns Array with [Neurons x Trails] of given day and stimulus
        """
        return self.dictionary[(day, stimulus)]

    def do_PCA(self, X):
        """
        does PCA on given data-set,
        returns list of coordinates [[x-values],[y-values],...] 
        """
        pca = PCA(n_components=self.dimension)
        pca.fit(X)
        return pca.components_.tolist(), sum(pca.explained_variance_ratio_)

    def create_df_singlestim(self, stimuli):
        """
        creates a pandas dataframe for single stimulus,
        !!! data needs to be converted into dicionary before that !!!
        """
        label = [(1, stimuli), (2, stimuli), (3, stimuli), (4, stimuli)]

        # Doing PCA for each day for one stimulus
        day1, var1 = self.do_PCA(self.dictionary[label[0]])
        day2, var2 = self.do_PCA(self.dictionary[label[1]])
        day3, var3 = self.do_PCA(self.dictionary[label[2]])
        day4, var4 = self.do_PCA(self.dictionary[label[3]])

        days = [day1, day2, day3, day4]
        for i in range(len(days)):
            for j in range(len(days[i])):
                days[i][j].insert(0, label[i])
        days = days[0] + days[1] + days[2] + days[3]

        cols = ['label']
        for i in range(self.dimension):
            cols.append('PC' + str((i+1)))
        self.dataframe = pd.DataFrame(days, columns=cols)

    def create_full_df(self):
        """
        creates a pandas dataframe for all stimuli 
        PCA or tSNE has been applied beforehand
        """
        if self.reduction_method == 'PCA':
            reduced_data, var = self.do_PCA(self.data)
            data = np.array([self.header, reduced_data[0], reduced_data[1]]).T
            cols = ['label']
            for i in range(self.dimension):
                cols.append('PC' + str((i+1)))

            print(len(labels))
            print(len(data), len(data[0]))
            self.dataframe = pd.DataFrame(data, columns=cols)
            print(self.dataframe)
            
        elif self.reduction_method == 'tSNE':
            pass
        else:
            print('Invalid reduction Method! Try PCA or tSNE!')

    def df_to_file(self, path):
        return 0

    def read_in_df(self, path):
        return 0


    def plot2D(self):
        # Cluster!!!
        # del plot am ende
        # https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489
        return 0

    def plot3D(self):
        return 0

    def minmax_scaler(self):
        return 0

    def normalization(self):
        return 0

    def standard_scaler(self):
        return 0


a = NeuralEarthquake_singlePopulation(
    "bl660-1_two_white_Pop01", "PCA", dimension=2)
a.read_population()
a.create_full_df()