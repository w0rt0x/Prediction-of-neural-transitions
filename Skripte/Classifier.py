from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd

class NeuralEarthquake_singlePopulation():
    
    def __init__(self, path, population):
        self.dataframe = pd.read_csv(path)
        self.population = population

    def read_in_df(self, path):
        self.dataframe = pd.read_csv(path)
