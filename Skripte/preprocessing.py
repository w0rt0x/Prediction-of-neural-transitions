import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import log as ln
import pandas as pd
import os
from os.path import isfile, join
from scipy.io import loadmat
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import plotly.express as px
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

class Preprocessor():

    def __init__(self, pop, NaN_replacement=0):
        """
        Takes in Path to directory and population name of file (.csv)
        """
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
        print(array)
        print(self.label.tolist())



pop = "bl693_no_white_Pop05"

a = Preprocessor(pop)