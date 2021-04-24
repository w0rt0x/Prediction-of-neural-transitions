import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
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
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


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
        data = np.array(data).T

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
        return pca.components_.T.tolist(), sum(pca.explained_variance_ratio_)

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
            for i in range(len(reduced_data)):
                reduced_data[i].insert(0, self.header[i])
            cols = ['label']
            for i in range(self.dimension):
                cols.append('PC' + str((i+1)))

            self.dataframe = pd.DataFrame(reduced_data, columns=cols)
            
        elif self.reduction_method == 'tSNE':
            pass
        else:
            print('Invalid reduction Method! Try PCA or tSNE!')

    def add_activity_to_df(self, path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\data_for_sam'):
        """
        Adds another column to the dataframe: response
        response-class is day/stimulux specific, 0 means no response
        """
        mat = loadmat(path + '\\' + self.population + '.mat')
        array = mat['cl_id']
        response = []
        
        for i in range(len(self.dataframe['label'])):
            day, stimulus = self.dataframe['label'][i][0], self.dataframe['label'][i][1]
            response.append(array[stimulus - 1, day - 1])

        self.dataframe['response'] = response

    def df_to_file(self, path):
        """
        writes Dataframe to .csv file
        """
        self.dataframe.to_csv(path + '\\{}.csv'.format(self.population))

    def read_in_df(self, path):
        self.dataframe = pd.read_csv(path)

    def get_df(self):
        return self.dataframe

    def plot2D_anim(self, day):
        """
        Plots 2D animated Scatter Plot
        """
        # Sources:
        # https://www.statology.org/matplotlib-scatterplot-color-by-value/
        # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
        # https://www.programcreek.com/python/example/102361/matplotlib.pyplot.pause
  
        plt.style.use('dark_background')
        plt.ion()
        plt.title("{}, Day {}".format(self.population, day))
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")
        plt.xlim([-0.05, 0.25])
        plt.ylim([-0.05, 0.25])
        plt.draw()
        colors = cm.plasma(np.linspace(0, 1, 34))
        for i in range(34):
            df = self.dataframe[self.dataframe.isin([(day, i)]).any(axis=1)]
            groups = df.groupby('label')
            for name, group in groups:
                plt.scatter(group.PC1, group.PC2, label=name, c=colors[i])
                plt.pause(0.3)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=4)
        plt.waitforbuttonpress()

    def plot2D(self, day, save=False, path=None):
        """
        Plots 2D Scatter Plot, can be saved to path
        """
        plt.style.use('dark_background')
        plt.title("{}, Day {}".format(self.population, day))
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")
        plt.xlim([-0.05, 0.25])
        plt.ylim([-0.05, 0.25])
        colors = cm.plasma(np.linspace(0, 1, 34))
        for i in range(34):
            df = self.dataframe[self.dataframe.isin([(day, i)]).any(axis=1)]
            groups = df.groupby('label')
            for name, group in groups:
                plt.scatter(group.PC1, group.PC2, label=name, c=colors[i], s=5)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=4)

        plt.subplots_adjust(bottom = 0.2)

        if save:
            plt.savefig(path, bbox_inches='tight', dpi=100)
        else:
            plt.show()
        
        plt.cla()

    def plot3D(self):
        fig = px.scatter_3d(self.dataframe, x='PC1', y='PC2', z='PC3',
                        color='label')

        fig.show()

    def minmax_scaler(self):
        return 0

    def normalization(self):
        return 0

    def standard_scaler(self):
        return 0


def plot_all_populations(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\2D_PCA_AllStimInOne', dim=2):
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
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))

        # Create Directory for all plotted Stimuli
        new_dir = destination + '\\' + str(pop)
        os.mkdir(new_dir)

        # read in data
        a = NeuralEarthquake_singlePopulation(pop, "PCA", dimension=dim)
        a.read_population()
        a.create_full_df()
        for day in range(1, 5):
            a.plot2D(day, True, new_dir + '\\' +'{}, Day {}.{}'.format(pop, day, 'png'))

a = NeuralEarthquake_singlePopulation(
    "bl687-1_no_white_Pop02", "PCA", dimension=20)
a.read_population()
a.create_full_df()
a.add_activity_to_df()
#a.plot2D_anim(4)
#a.plot2D(4, True, r'C:\Users\Sam\Desktop\bl684_no_white_Pop11.png')
a.df_to_file(r"C:\Users\Sam\Desktop")
