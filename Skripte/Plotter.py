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
        Plots 2D Data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
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
            plt.scatter(x, y, c=label, alpha=0.3)
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
            
            plt.clf()
            plt.cla()
            plt.close()

    def plot_3D(self, method: str, x_axis: str, y_axis: str, z_axis: str, show: bool=True, dest_path: str=None):
        """
        Plots 2D Data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param z_axis (str) - name of z-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
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
            ax.scatter(x, y, z, c=label, alpha=0.5)
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

            plt.clf()
            plt.cla()
            plt.close()

    def compare_n_neurons(self, title: str, neurons:list=list(range(5, 101, 5))):
        """
        Compares performance of n most active neurons
        :param title (str) - Title of plot
        :param neurons (list) - List of integers that are directory names of path in init function
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
        
    def plot_actual_vs_predicted(self, method: str, x_axis: str, y_axis: str, show: bool=True, dest_path: str=None):
        """
        Plots 2 Plots: Predicted(SVM) vs Actual data
        :param methods (str) - Can be PCA, n most active neurons, t-SNE
        :param x_axis (str) - name of x-axis
        :param y_axis (str) - name of y-axis
        :param show (bool) - dafault is true, shows plot when done
        :param dest_path (str) - default is None, if its not none the plot will be saved to that directory
        """
        for pop in self.populations:
            c = Classifier([pop], self.path)
            c.split_trial_wise()
            c.use_SMOTE()
            c.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            X, x, Y, y = c.get_data()
            x = x.T

            y = self.__multiclass_to_color(y.tolist())
            pred = self.__multiclass_to_color(c.get_predictions().tolist())

                # Plotting the Data
            figure, axis = plt.subplots(1, 2, figsize=(13,7))
            axis[0].scatter(x[0], x[1], c=y, alpha=0.5)
            axis[0].set_title("Actual Data")
            axis[0].set_xlabel(x_axis)
            axis[0].set_ylabel(y_axis)
                
            # For Cosine Function
            axis[1].scatter(x[0], x[1], c=pred, alpha=0.5)
            axis[1].set_title("Prediction")
            axis[1].set_xlabel(x_axis)
            axis[1].set_ylabel(y_axis)

            yellow = mpatches.Patch(color='yellow', label='1->1')
            red = mpatches.Patch(color='red', label='1->0')
            green = mpatches.Patch(color='green', label='0->1')
            cyan = mpatches.Patch(color='cyan', label='0->0')
            figure.legend(handles=[yellow, red, green, cyan])

            figure.suptitle("{} - Test-Data of {},\n SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')\n\n".format(method, pop))
            

            if show:
                plt.show() 
            if dest_path !=None:
                plt.savefig(dest_path + '\\{}.png'.format(self.populations[0]))

            plt.clf()
            plt.cla()
            plt.close()

    def __multiclass_to_color(self, label):
        for i in range(len(label)):
                if label[i] == '0->0':
                    label[i] = 'cyan'
                if label[i] == '1->0':
                    label[i] = 'red'
                if label[i] == '0->1':
                    label[i] = 'green'
                if label[i] == '1->1':
                    label[i] = 'yellow'
        return label

    def compare_models(self, pops: list, paths:list, x_axis:str, title:str, width: float=0.2, show:bool=True, dest_path:str=None, names = None):
    # names ist list of string
    # random automatisch
        micro = []
        macro = []
        weighted = []
        pops.append(self.populations)
        paths.append(self.path)
        if names == None:
            names = []
            for i in range(len(pops)):
                names.append('\n'.join(pops[i]))
                names.append('\n'.join(pops[i]) + '\n(random labels)')
        
        for i in range(len(pops)):
            c = Classifier(pops[i], paths[i])
            c.split_trial_wise()
            c.use_SMOTE()
            c.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = c.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])
                
            # Randomized labels
            r = Classifier(self.populations, self.path)
            r.split_trial_wise()
            r.use_SMOTE()
            r.shuffle_labels()
            r.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')
            report = r.get_report()
            macro.append(report['macro avg']['f1-score'])
            micro.append(report['accuracy'])
            weighted.append(report['weighted avg']['f1-score'])

        x = np.arange(0, len(names))
        plt.bar(x, micro, width=width, color='yellow', label="Micro F1-Score")
        plt.bar(x + width, macro, width=width, color='blue', label="Macro F1-Score")
        plt.bar(x + 2*width, weighted, width=width, color='green', label="Weighted F1-Score")
        plt.xticks(x, names)
        plt.ylabel("Scores")
        plt.xlabel(x_axis)
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.legend()
        plt.title(title)
        if show:
            plt.show()
        if dest_path !=None:
            plt.savefig(dest_path + '\\{}.png'.format(title))

        plt.clf()
        plt.cla()
        plt.close()


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
a.compare_models([['bl693_no_white_Pop02', 'bl693_no_white_Pop03', 'bl693_no_white_Pop05']], [r'D:\Dataframes\tSNE\3D_perp30'], "Input Populations", "Prediction of Multiclass Labels with \n SVM(rbf Kernel, balanced class weights)\n and t-SNE 3D Data")
#a.plot_2D("t-SNE", "t-SNE Component 1", "t-SNE Component 2")
#a.plot_actual_vs_predicted("t-SNE", "Component 1", "Component 2")
#b = Plotter(get_all_pop(), r'D:\Dataframes\tSNE\perp30')
#b.plot_actual_vs_predicted("t-SNE", "t-SNE Component 1", "t-SNE Component 2", show=False, dest_path=r'D:\Dataframes\tSNE\2D_actual_vs_predicted')