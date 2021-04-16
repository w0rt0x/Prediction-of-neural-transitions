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
    print("Input: ", len(X), len(X[0]))
    pca.fit(X)
    # Getting Variance, Title and data-points
    var = sum(pca.explained_variance_ratio_)
    title = "{}: {} components have {}% of the variance ({} Standard-Scaler)".format(label, components, round(var * 100, 2), used)
    print("PC's: ", len(pca.components_), len(pca.components_[0]))
    return pca.components_.tolist(), title
    

if __name__ == "__main__":
    
    header, data = get_data(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_class.mat",
                            r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_lact.mat")

    replace_nan(data, 0)
    df = data_to_dict(data, header)

    print(len(df[(1, 1)]), len(df[(1, 1)][0]))
    points, title = do_PCA(df[(1,1)], 2, (1,1))
    d, title = do_PCA(df[(2,1)], 2, (2,1))
    d2, title = do_PCA(df[(3,1)], 2, (3,1))
    d3, title = do_PCA(df[(4,1)], 2, (4,1))
    """
    l = [(1, 1),(2, 1),(3, 1),(4, 1)]
    df = [points, d, d2, d3]
    
    for i in range(len(df)):
        for j in range(len(df[i])):
            df[i][j].insert(0, l[i])
    df = df[0] + df[1] + df[2] + df[3] 
    df = pd.DataFrame(df, columns = ['label', 'PC1', 'PC2', 'PC3'])
    print(df)
    """
    plot2D([points, d, d2, d3], "2D-PCA for Day 1-4, Stimulus 1")
    #plot3D(df, "3D-PCA for Day 1-4, Stimulus 1")
    
    
# Sortieren: Days und Trials in Dict - (day, Trial) als Key
# Alle 4 Tage des gleichen Trials in 1 Plot

