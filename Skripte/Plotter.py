import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy.io import loadmat
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def get_data(path_class, path_lact):
    """
    Reads in matlab files,
    returns header list and neuron data
    """
    # Getting labels
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
    # Swapping is done with .T from numpy: https://note.nkmk.me/en/python-list-transpose/
    return np.array(header), np.array(data).T


def replace_nan(data, replacement):
    """ 
    replaces nan with replacement in data-matrix,
    does not return list because lists are matuable
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            if math.isnan(data[i][j]):
                data[i][j] = replacement

def data_to_dict(data, header):
    """
    Returns Dictionary with headers as keys that hold corresponding data,
    e.g. "(Day 4, Trial 7)": [Neurons x Trials]
    """
    d = dict()

    for i in range(len(header)):
        if tuple(header[i]) in d:
            # Tuple conversion because numpy-arrays can't be hashed in a dictionary
            d[tuple(header[i])] = d[tuple(header[i])] + [data[i]]

        else:
            d[tuple(header[i])] = [data[i]]

    # Transpose Matrix, so that Neurons x Trials
    keys = list(d.keys())
    for k in keys:
        d[k] = np.array(d[k]).T
    return d

def plot2D(df, title):
    """
    Plots 2D Scatter Plot for PCA
    Same stimulus, for all 4 days
    Source:
    https://www.statology.org/matplotlib-scatterplot-color-by-value/
    """
    plt.style.use('dark_background')
    colors = ["aqua", "lime", "deeppink", "darkorange"]
    groups = df.groupby('label')
    
    for name, group in groups:
        plt.scatter(group.PC1, group.PC2, label=name, c=colors[name[0] - 1])

    plt.title(title)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
    plt.show()

def plot3D(df, text):
    # https://plotly.com/python/3d-scatter-plots/
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
              color='label', )

    # Label
    # https://stackoverflow.com/questions/26941135/show-legend-and-label-axes-in-plotly-3d-scatter-plots
    fig.show()


def do_PCA(X, components, label, scaler=False):
    """
    returns reduced x-axis!
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 
    https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb
    """
    # Optional: Use standard sclaler
    used = "without"
    if scaler:
        """
        Standardize features by removing the mean and scaling to unit variance:
        z = (x - u) / s, where
        z is scaled data,
        x is to be scaled data,
        u is the mean of the training samples and
        s is the standard deviation of the training samples
        """
        X = StandardScaler().fit_transform(X)
        used = "with"

    # Starting PCA
    pca = PCA(n_components=components)
    pca.fit(X)

    # Getting Variance, Title and data-points
    var = sum(pca.explained_variance_ratio_)
    title = "{}: {} components have {}% of the variance ({} Standard-Scaler)".format(label, components, round(var * 100, 2), used)

    return pca.components_.T.tolist(), title

def crate_dataframe(data_dic, stimuli, dim):
    """
    creates pandas dataframe with Principle Components for all 4 days for a given stimulus
    """
    label = [(1, stimuli),(2, stimuli),(3, stimuli),(4, stimuli)]

    # Doing PCA for each day for one stimulus
    day1, title = do_PCA(data_dic[label[0]], dim, label[0])
    day2, title = do_PCA(data_dic[label[1]], dim, label[1])
    day3, title = do_PCA(data_dic[label[2]], dim, label[2])
    day4, title = do_PCA(data_dic[label[3]], dim, label[3])

    days = [day1, day2, day3, day4]
    for i in range(len(days)):
        for j in range(len(days[i])):
            days[i][j].insert(0, label[i])
    days = days[0] + days[1] + days[2]+ days[3]

    cols = ['label']
    for i in range(dim):
        cols.append('PC' + str((i+1)))
    return pd.DataFrame(days, columns = cols)
    


if __name__ == "__main__":
    stimulus = 1
    dimension = 3
    population = "bl660-1_two_white_Pop01" #"bl660-1_two_white_Pop01"


    header, data = get_data(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_class.mat".format(population),
                            r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_lact.mat".format(population))

    replace_nan(data, 0)
    dictionary = data_to_dict(data, header)
    df = crate_dataframe(dictionary, stimulus, dimension)
    
    if dimension == 2:
        plot2D(df, "{}: 2D-PCA for Day 1-4, Stimulus {}".format(population, stimulus))
    elif (dimension == 3):
        plot3D(df, "{}: 3D-PCA for Day 1-4, Stimulus {}".format(population, stimulus))
    else:
        print(df)
    

