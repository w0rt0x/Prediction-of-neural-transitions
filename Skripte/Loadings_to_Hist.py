import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
from os.path import isfile, join

def plot_histo(X, Y, title1, title2, n_bins=500):
    """ 
    Plots 2 histos next to each other
    X schould a with a response
    Y should have no response
    """
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(X, bins=n_bins)
    axs[1].hist(Y, bins=n_bins)
    axs[0].title.set_text(title1)
    axs[1].title.set_text(title2)
    plt.show()

def create_Histos(path=r'D:\Dataframes\Loadings'):
    """ 
    This function was used to plot histograms of the mean or total sum of loadings
    in order to check if the response-classes are distinguishable
    """
    means_active = []
    sums_active = []
    means_not_active = []
    sums_not_active = []

    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for f in files:
        df = pd.read_csv(path + '\\' + f)
        mean = df['loadings_mean'].tolist()
        total_sum = df['loadings_sum'].tolist()
        response = df['response'].tolist()

        for i in range(len(response)):
            if response[i]>0:
                means_active.append(mean[i])
                sums_active.append(total_sum[i])
            else:
                means_not_active.append(mean[i])
                sums_not_active.append(total_sum[i])
    
     # Mean Plot
    plot_histo(means_active, means_not_active, 'Mean of Loadings with response,\n {} in total'.format(len(means_active)), 'Mean of Loadings without response,\n {} in total'.format(len(means_not_active)))
    plot_histo(sums_active, sums_not_active, 'Sum of Loadings with response,\n {} in total'.format(len(sums_active)), 'Sum of Loadings without response,\n {} in total'.format(len(sums_not_active)))    

if __name__ == '__main__':
    #create_Histos(r'D:\Dataframes\Loadings_StandardScaler')
    #create_Histos(r'D:\Dataframes\Loadings_MinMax')
    create_Histos(r'D:\Dataframes\Loadings_Norm')