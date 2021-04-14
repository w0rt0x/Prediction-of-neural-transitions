import seaborn as sns
import matplotlib as plt
import numpy as np
import pandas as pd
import csv
from scipy.io import loadmat
import math


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
    return header, np.array(data).T.tolist()


def replace_nan(data, replacement):
    """ 
    replaces nan with replacement in data-matrix,
    does not return list because lists are matuable
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            if math.isnan(data[i][j]):
                data[i][j] = replacement

def plot_2D_PCA():
#inner function??
# Dimension als parameter??


if __name__ == "__main__":
    header, data = get_data(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_class.mat",
                            r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\bl660-1_two_white_Pop01_lact.mat")
    replace_nan(data, 0)

