import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os
from os.path import isfile, join
from Classifier import Classifier

class Plotter:

    def __init__(self, populations: list, path:str):
        """
        Needs List of populations and directory with dataframes
        :param populations - List of populationnames, without .csv at the end
        :param path - string path to directory
        """
        self.populations = populations
        self.path = path
        full_paths = []
        for i in range(len(populations)):
            full_paths.append(path + '\\' + populations[i] + '.csv')
        self.full_paths = full_paths

    def plot_2D(self, method: str, x_axis: str, y_axis: str, show: bool=True, dest_path: str=None):
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
            plt.scatter(x, y, c=label, alpha=0.5)
            name = self.populations[self.full_paths.index(pop)]
            plt.title('{} of {}'.format(method, name))
            
            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            plt.legend(handles=[yellow, red, green, cyan])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            if show:
                plt.show()
            if dest_path != None:
                plt.savefig(dest_path + '\\{}.png'.format(name))

    def plot_3D(self, method: str, x_axis: str, y_axis: str, z_axis: str, show: bool=True, dest_path: str=None):
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
            ax.scatter(x, y, z, c=label, alpha=0.7)
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
            if show:
                plt.show()
            if dest_path != None:
                plt.savefig(dest_path + '\\{}.png'.format(name))

    def compare_n_neurons(self, title: str, neurons:list=list(range(5, 101, 5))):
        """
        Compares 
        """
        macro = []
        micro = []
        weighted = []
        macro_r = []
        micro_r = []
        weighted_r = []

        for n in neurons:
            # Regular run
            c = Classifier(self.populations, self.path + str(n))
            c.split_trial_wise()
            c.use_SMOTE()
            c.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = c.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])
            
            # Randomized labels
            r = Classifier(self.populations, self.path + str(n))
            r.split_trial_wise()
            r.use_SMOTE()
            r.shuffle_labels()
            r.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = r.get_report()
            macro_r.append(report['macro avg']['f1-score'])
            micro_r.append(report['accuracy'])
            weighted_r.append(report['weighted avg']['f1-score'])

        
        plt.plot(neurons,macro, marker = 'o', color='#f70d1a', label="Macro F1")
        plt.plot(neurons,micro, marker = 'x', color='#08088A', label="Micro F1")
        plt.plot(neurons,weighted, marker = '+', color='#FFBF00', label="weighted F1")
        plt.plot(neurons,macro_r, marker = 'o', color='#f70d1a', label="Macro F1 (random)", linestyle = '--')
        plt.plot(neurons,micro_r, marker = 'x', color='#08088A', label="Micro F1 (random)", linestyle = '--')
        plt.plot(neurons,weighted_r, marker = '+', color='#FFBF00', label="weighted F1 (random)", linestyle = '--')
        plt.xlabel("#Neurons")
        plt.xticks(neurons)
        plt.ylabel("F1-Scores")
        plt.ylim([0, 1])
        plt.legend(loc="upper left")
        plt.title(title)
        plt.show()
        

def get_all_pop(path: str=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'):
    """
    returns all population-names
    """
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])
    return list(populations)

# removing dubs
a = Plotter(['bl693_no_white_Pop05'], r'D:\Dataframes\tSNE\3D_perp30')
a.plot_3D("t-SNE", "Component 1", "Component 2", "Component3")
#a.compare_n_neurons("bl693_no_white_Pop05, bl693_no_white_Pop02, bl693_no_white_Pop03,\n Smote on Training-Data,\n SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')")
#a.plot_2D('t-SNE Plot', '1st Component', '2nd Component', r'C:\Users\Sam\Desktop')