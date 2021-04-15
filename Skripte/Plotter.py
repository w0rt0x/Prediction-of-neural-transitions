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
#from mpl_toolkits.mplot3d import Axes3D

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

def plot2D(data, title):
    """
    Plots 2D Scatter Plot for PCA
    Same tone, for all 4 days
    """
    plt.style.use('dark_background')
    plt.scatter(data[0][0], data[0][1], color= "aqua", label = "Day 1")
    plt.scatter(data[1][0], data[1][1], color= "lime", label = "Day 2")
    plt.scatter(data[2][0], data[2][1], color= "deeppink", label = "Day 3")
    plt.scatter(data[3][0], data[3][1], color= "darkorange", label = "Day 4")
    plt.title(title)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")

    # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
    plt.show()

def plot3D2(data):
    # https://plotly.com/python/3d-scatter-plots/
    fig = px.scatter_3d(x=data[0], y=data[1], z=data[2], color="red")
    fig.show()


def do_PCA(X, components, label, scaler=False):
    """
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
    return pca.components_, title
    

if __name__ == "__main__":
    
    header, data = get_data(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_class.mat",
                            r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_lact.mat")
    replace_nan(data, 0)
    points, title = do_PCA(data[0:30].T, 3, header[0])
    d, title = do_PCA(data[30:60].T, 3, header[1])
    d2, title = do_PCA(data[60:90].T, 3, header[2])
    d3, title = do_PCA(data[90:120].T, 3, header[3])
    #plot2D([points, d, d2, d3], "2D-PCA for Day 1-4, Stimulus 1")
    plot3D2(points)
    #plot3D([points, d, d2, d3], "3D-PCA for Day 1-4, Stimulus 1")
    
    
# Sortieren: Days und Trials in Dict - (day, Trial) als Key
# Alle 4 Tage des gleichen Trials in 1 Plot

