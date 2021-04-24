from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np


class NeuralEarthquake_Classifier():

    def __init__(self, path, population):
        self.dataframe = pd.read_csv(path)
        self.population = population
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self. accuracy = None

    def read_in_df(self, path):
        self.dataframe = pd.read_csv(path)

    def prepare_binary_labels(self, split_ratio=0.2, randomState=None):
        """
        Converts dataframe into list with PC's and labels,
        labels are 0 (no activity) or 1 (aktivity > 0).
        saves X_train, y_train, X_test, y_test as class attributes and sets dataframe to None after.
        Optional variable split_ratio sets ratio for training/test split, bist be between 0 and 1.
        randomState must be int, for reproducable outcomes
        """
        X = []
        y = []

        for index, row in self.dataframe.iterrows():
            X.append(row[2:-1].tolist())
            y.append(row[-1])
        
        self.dataframe = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split_ratio, random_state=randomState)
        X, y = None, None
        print(len(self.X_train), len(self.X_test))
        print(len(self.y_train), len(self.y_test))

    def do_Logistic_Regression(self):
        """
        performs logistic regression 
        """
        # Weitere Parameter
        return 0

    def do_SVM(self, kernel="linear", c=1):
        """
        performs Support Vectors Machine on dataset
        """
        return 0

    def plot_CM(self, title=None, path=None):
        """
        plots Confusion Matrix of results, can be saved to path
        """
        # accuracy und population in title
        return 0

a = NeuralEarthquake_Classifier(r"C:\Users\Sam\Desktop\bl687-1_no_white_Pop02.csv", 'bl687-1_no_white_Pop02')
a.prepare_binary_labels()