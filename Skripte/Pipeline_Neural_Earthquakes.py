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

"""
This Class/Script is mostly for data-preprocessing. The Data is given in matlab-files, which are converted to
pandas dataframes after performing a dimensionality reduction (PCA) on it. The Class also offers the Option
to plot the data, if the dimensions (d=2) are right.
"""



class NeuralEarthquake_singlePopulation():

    def __init__(self, population, reduction_method='PCA', dimension=20, path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', replacement=0):
        self.dimension = dimension
        self.population = population
        self.reduction_method = reduction_method
        self.path_class = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_class.mat".format(
            self.population)
        self.path_lact = r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_lact.mat".format(
            self.population)
        self.data = None
        self.header = None
        self.trials = None
        self.dataframe = None
        self.replacement = replacement  # replacement for NaN
        self.loading_matrix = None

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

    def get_most_active_neurons(self, n=30):
        """gets the n most active neurons and saves them in a dataframe"""
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

        neurons = np.array(neurons).T
        neurons = neurons.tolist()

        for i in range(len(neurons)):
            neurons[i].insert(0, self.header[i])
            cols = ['label']
            for i in range(n):
                cols.append('N' + str((i+1)))

        self.dataframe = pd.DataFrame(neurons, columns=cols)

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

        return d

    def do_PCA(self, X):
        """
        does PCA on given data-set,
        returns list of coordinates [[x-values],[y-values],...] 
        """
        pca = PCA(n_components=self.dimension)
        pca.fit(X)
        # also calculating loadings, Source:
        # https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
        self.loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)

        return pca.components_.T.tolist(), sum(pca.explained_variance_ratio_)

    def do_isomap(self, X):
        """
        does isomap on given data-set
        """
        # Source: https://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/
        iso = Isomap(n_components=self.dimension)
        iso.fit(X)
        manifold_2Da = iso.transform(X)
        return manifold_2Da.tolist()

    def get_loading_data(self, path=r'D:\Dataframes\Loadings'):
        """
        Calculates each Loading matrix (Trial x PCs) and their mean/total-sum value,
        saves dataframe to path
        """
        data = []
        loadings_sum = []
        loadings_mean = []
        response = []
        for day in range(1, 5):
            for stim in range(1, 35):
                # Getting indices for loading-matrix
                indices = [i for i, x in enumerate(self.header) if x == (day, stim)]
                data.append((day, stim))
                # Calculating mean and sum
                loadings_sum.append(self.loading_matrix[indices[0]:indices[-1]].sum())
                loadings_mean.append(self.loading_matrix[indices[0]:indices[-1]].mean())
                response.append(self.dataframe.response[indices[0]])

        df = pd.DataFrame(list(zip(data, loadings_mean, loadings_sum, response)), columns =['day,stim', 'loadings_mean', 'loadings_sum', 'response'])
        df.to_csv(path + '\\Loadings_{}.csv'.format(self.population))

    def get_min_max(self):
        """
        saves max and min loading for plotting
        """
        vals = []
        for k in self.loading_matrix:
            for i in k:
                vals.append(i)
        self.max = max(vals)
        self.min = min(vals)

    def create_full_df(self):
        """
        creates a pandas dataframe for all stimuli 
        PCA or tSNE or isomap has been applied beforehand
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
        elif self.reduction_method == 'isomap':
            reduced_data = self.do_isomap(self.data.T)
            for i in range(len(reduced_data)):
                reduced_data[i].insert(0, self.header[i])
            cols = ['label']
            for i in range(self.dimension):
                cols.append('PC' + str((i+1)))

            self.dataframe = pd.DataFrame(reduced_data, columns=cols)
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

    def make_transition_labels(self, binary=True):
        """
        changes the Response labels for binary transitions
        """
        response = []
        neurons = []
        trails = []
        matrix = self.dataframe.to_numpy()
        for i in range(len(matrix)):
            trails.append(matrix[i][0])
            response.append(1 if (matrix[i][-1] > 0) else 0)
            neurons.append(matrix[i][1:-1])

        transistions = [0] * len(response)
        labels = set(trails)
        for trail in labels:
            # Skipping Day 4
            if trail[0] == 4:
                pass
            else:
                if binary:
                    next_day = (trail[0] + 1, trail[1])
                    # Binary Transition: from 0 to response
                    if response[trails.index(next_day)] > 0 and response[trails.index(trail)] == 0:
                        indices = [i for i in range(len(trails)) if trails[i] == trail]
                        for j in indices:
                            transistions[j] = 1
                else:
                    #Multiclass
                    next_day = (trail[0] + 1, trail[1])
                    indices = [i for i in range(len(trails)) if trails[i] == trail]
                    # Binary Transition: from 0 to response
                    label = '-'
                    if response[trails.index(next_day)] == 0 and response[trails.index(trail)] == 0:
                        label = '0->0'
                    if response[trails.index(next_day)] > 0 and response[trails.index(trail)] > 0:
                        label = '1->1'
                    if response[trails.index(next_day)] > 0 and response[trails.index(trail)] == 0:
                        label = '0->1'
                    if response[trails.index(next_day)] == 0 and response[trails.index(trail)] > 0:
                        label = '1->0'
                    for j in indices:
                        transistions[j] = label

        self.dataframe['response'] = transistions

    def df_to_file(self, path):
        """
        writes Dataframe to .csv file
        """
        self.dataframe.to_csv(path + '\\{}.csv'.format(self.population))

    def read_in_df(self, path):
        """
        reads in dataframe from .csv file
        """
        self.dataframe = pd.read_csv(path)

    def get_df(self):
        """
        returns dataframe
        """
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
                response = list(df[df['label']==name]['response'])
                if response[0] > 0:
                    plt.scatter(group.PC1, group.PC2, label=name, c=colors[i], marker='x')
                else:
                    plt.scatter(group.PC1, group.PC2, label=name, c=colors[i], marker='o')
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

    def plot2D_loadings(self, day, stim, save=False, path=None):
        """
        Plots PC's per stimulus with loadings and activity
        """
        indices = [i for i, x in enumerate(self.header) if x == (day, stim)]
        plt.style.use('dark_background')
        plt.title("Loadings of {}, Day {}, Stimulus {}".format(self.population, day, stim))
        plt.xlabel("PC's")
        plt.ylabel("Trials of Day{}, Stimulus {}".format(day, stim))
        plt.xticks(range(1, 22))
        plt.yticks(range(1, len(indices)))
        plt.pcolor(self.loading_matrix[indices[0]:indices[-1]], cmap='plasma', vmax=self.max, vmin=self.min)
        cb = plt.colorbar()

        if save:
            plt.savefig(path)
        else:
            plt.show()
        
        plt.cla()
        cb.remove()

    def minmax_scaler(self):
        self.data = MinMaxScaler().fit_transform(self.data)

    def normalization(self):
        self.data = Normalizer().fit_transform(self.data)

    def standard_scaler(self):
        self.data = StandardScaler().fit_transform(self.data)

    def apply_log(self):
        """applys log on data"""
        self.data = ln(self.data)


def plot_all_populations(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\2D_PCA_AllStimInOne', dim=2):
    """
    Creates Plots for all given matlab files after performing PCA on them
    """
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

def create_all_df(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'D:\Dataframes\20PCs', dim=20):
    """
    Uses neuralEarthquake class to transform all matlab files into pandas dataframes,
    which are savind in .csv files
    """
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

        # read in data
        a = NeuralEarthquake_singlePopulation(pop, reduction_method='PCA', dimension=dim)
        a.read_population()
        a.create_full_df()
        a.add_activity_to_df()
        a.df_to_file(destination)
        a = None

def plot_all_loadings(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Loading_Plots', dim=20):
    """
    Creates Loadings plots for all files
    """
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
        a.get_min_max()

        for day in range(1, 5):
            for stim in range(1, 35):
                a.plot2D_loadings(day, stim, True, new_dir + '\\{},Day{},stim{}.png'.format(pop, day, stim))

def merge_all_df(directory = r'D:\Dataframes\20PCs', destination=r'D:\Dataframes\merged_20PCs.csv'):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    final_df = pd.concat([pd.read_csv(f) for f in files])
    final_df.to_csv(destination, index=False)

def create_all_loadings(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'D:\Dataframes\Loadings', dim=20):
    """
    Creates all files that hold the mean and total sum of loadings for all populations
    """
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

        # read in data
        a = NeuralEarthquake_singlePopulation(pop, "PCA", dimension=dim)
        a.read_population()
        a.normalization()
        a.create_full_df()
        a.add_activity_to_df()
        a.get_loading_data(destination)

def extract_all_neurons(n=30, path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'D:\Dataframes\30_mostActive_Neurons', binary=True):
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

        # read in data
        a = NeuralEarthquake_singlePopulation(pop)
        a.read_population()
        a.get_most_active_neurons(n=n)
        a.add_activity_to_df()
        a.make_transition_labels(binary=binary)
        a.df_to_file(destination)
        a = None
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))

def pca_multi(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'D:\Dataframes\30_mostActive_Neurons', binary=True):
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

        # read in data
        a = NeuralEarthquake_singlePopulation(pop)
        a.read_population()
        a.create_full_df()
        a.add_activity_to_df()
        a.make_transition_labels(binary=binary)
        a.df_to_file(destination)
        a = None
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))

def do_isomap(path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten', destination=r'D:\Dataframes\30_mostActive_Neurons', binary=True, dim=2):
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

        # read in data
        a = NeuralEarthquake_singlePopulation(pop, reduction_method='isomap', dimension=dim)
        a.read_population()
        a.create_full_df()
        #a.add_activity_to_df()
        a.make_transition_labels(binary=binary)
        a.df_to_file(destination)
        a = None
        print("{} of {} done".format(populations.index(pop) + 1, len(populations)))

do_isomap(destination=r'D:\Dataframes\isomap_multi', binary=False)
#a = NeuralEarthquake_singlePopulation("bl693_no_white_Pop05", "PCA", dimension=20)
#a.read_population()
#a.get_most_active_neurons()
#a.standard_scaler()
#a.create_full_df()
#a.add_activity_to_df()
#a.make_transition_labels()
#a.get_loading_data()
#a.plot2D_loadings(2, 5, True, r'C:\Users\Sam\Desktop')
#a.plot2D_anim(1)
#a.plot2D(4, True, r"C:\Users\Sam\Desktop\bl684_no_white_Pop11.png")
#a.df_to_file(r"C:\Users\Sam\Desktop")
#create_all_loadings(destination=r'D:\Dataframes\Loadings_Norm')
#create_all_df(destination=r"D:\Dataframes\20PCs_withLog")