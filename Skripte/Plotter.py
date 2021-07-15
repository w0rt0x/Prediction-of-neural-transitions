import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os
from os.path import isfile, join

class Plotter:

    def __init__(self, populations: list, path:str):
        """
        
        :param populations - List of populationnames, without .csv at the end
        :param path - string path to directory
        """
        self.populations = populations
        self.path = path
        full_paths = []
        for i in range(len(populations)):
            full_paths.append(path + '\\' + populations[i] + '.csv')
        self.full_paths = full_paths

    def plot_2D(self, method: str, x_axis: str, y_axis: str, dest_path: str):
        """
        
        """
        for pop in self.full_paths:
            df = pd.read_csv(pop)
            # Getting Data
            header = df['label'].tolist()
            x = df['Component 1'].tolist()
            y = df['Component 2'].tolist()
            label = df['response'].tolist()
            #Removing Day 4
            i = header.index('(4, 1)')
            x = x[:i]
            y = y[:i]
            label = label[:i]
            for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'yellow'

            # Plotting
            plt.scatter(x, y, c=label)
            name = self.populations[self.full_paths.index(pop)]
            plt.title('{} of {}'.format(method, name))
            
            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            plt.legend(handles=[yellow, red, green, cyan])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.show()
            #plt.savefig(dest_path + '\\{}.png'.format(name))

    def plot_3D(self, method: str, x_axis: str, y_axis: str, z_axis: str, dest_path: str):
        """
        """
        for pop in self.full_paths:
            fig = plt.figure()
            ax = plt.axes(projection ='3d')
            df = pd.read_csv(pop)
            # Getting Data
            header = df['label'].tolist()
            x = df['Component 1'].tolist()
            y = df['Component 2'].tolist()
            z = df['Component 3'].tolist()
            label = df['response'].tolist()
            #Removing Day 4
            i = header.index('(4, 1)')
            x = x[:i]
            y = y[:i]
            z = z[:i]
            label = label[:i]
            for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'yellow'

            # Plotting
            ax = plt.axes(projection ="3d")
            ax.scatter(x, y, z, c=label)
            name = self.populations[self.full_paths.index(pop)]
            ax.set_title('{} of {}'.format(method, name))
            
            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            plt.legend(handles=[yellow, red, green, cyan])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_zlabel(z_axis)
            #plt.show()
            plt.savefig(dest_path + '\\{}.png'.format(name))

path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'
populations = set()
files = [f for f in os.listdir(path) if isfile(join(path, f))]
for i in files:
    if "_class.mat" in i:
        populations.add(i[:-10])

    if "_lact.mat" in i:
        populations.add(i[:-9])

# removing dubs
populations = list(populations)
a = Plotter(['bl693_no_white_Pop07'], r'D:\Dataframes\tSNE\perp30')
a.plot_2D('t-SNE Plot', '1st Component', '2nd Component', r"D:\Dataframes\most_active_neurons\2D Plots")