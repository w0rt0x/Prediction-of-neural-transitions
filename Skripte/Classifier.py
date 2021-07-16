from sklearn.svm import SVC
import os
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns
from random import sample, shuffle
from copy import deepcopy
from imblearn.over_sampling import ADASYN, SMOTE
from typing import Tuple


class Classifier():

    def __init__(self, populations: list, direc: str):
        """
        :param populations(list) - List with Population names
        :param direc - path(str) to directory with data
        """
        self.populations = populations
        self.dataframes = []
        for i in range(len(populations)):
            self.dataframes.append(pd.read_csv(direc + '\\{}.csv'.format(populations[i])))

    def random_split(self, split_ratio: float=0.2, randomState: int=None, remove_day4: bool=True):
        """
        Taking random samples for training/Test with the train_test_split function by Scikit learn
        for each population
        :param Split-ratio (float) - Ratio of Training/Test Split
        :param randomState (int) - Seed
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        """
        X = []
        y = []

        for df in self.dataframes:
            for index, row in df.iterrows():
                if remove_day4 and eval(row[1])[0] != 4:
                    X.append(row[2:-1].tolist())
                    y.append(row[-1])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=randomState)
        X, y = None, None

    def split_population_wise(self, n:int, remove_day4: bool=True):
        """
        This splitter-function takes n random populations for testing and all remaining for training.
        Those n populations are taken from the provided list at the beginning (init)
        :param n (int) - number of populations used for training
        :param remove_day4 (bool) - removes day4 trials, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = []      
        
        # Splitting
        dfs = deepcopy(self.dataframes)
        shuffle(dfs)
        df_test = dfs[:n]
        df_train = dfs[n:]

        for df in df_train:
            X, x, Y, y = self.__split_df(df, 0.0, remove_day4, False)
            X_train = X_train + X
            y_train = y_train + Y

        for df in df_test:
            X, x, Y, y = self.__split_df(df, 0.0, remove_day4, False)
            X_test = X_test + X
            y_test = y_test + Y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

    def print_shape(self):
        """
        Prints shape of training/test data
        """
        print("X_train: ", self.X_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_train: ", self.y_train.shape)
        print("y_test: ", self.y_test.shape)

    def split_trial_wise(self, split_ratio: float=0.2, remove_day4: bool=True, shuffle: bool=True):
        """
        Each Population has ~20 repetitions per trial. This function splits each of those repetitions so that 
        training and test data have some repetitions (split_ratio)
        :param Split-ratio (float) - Ratio of Training/Test Split, default is 0.2
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

    def __split_df(self, df:pd.DataFrame, ratio:float, rem_day4:bool, shuffle:bool) -> Tuple[list, list, list, list]:
        """
        returns Training/Test data as lists
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

        header = set(df['label'].tolist())
        # Removing Day 4
        trails = set()
        for i in header:
            trail = eval(i)
            if trail[0] != 4 and rem_day4:
                trails.add(i)
            else:
                if trail[0] == 4 and not(rem_day4):
                    trails.add(i)

        header = trails

        # Getting all the matrices from the trials
        print(len(df['label'].tolist()))
        for trial in header:
            # geting rows with (day, Trail)-label
            rows = df.loc[df['label'] == trial].to_numpy()
            # getting response label
            response = rows[0][-1]
            # getting the actual data from the matrix
            rows = np.delete(rows, np.s_[0,1,-1], axis=1)
            if shuffle:
                # shuffle PC-Matrix
                np.random.shuffle(rows)

            # Splitting into Test and training
            cut = int(ratio*len(rows))
            for i in range(len(rows)):
                if i < cut or ratio == 0.0:
                    X_train.append(rows[i])
                    y_train.append(response)
                else:
                    X_test.append(rows[i])
                    y_test.append(response)

        return X_train, X_test, y_train, y_test

    def split_transitions(self, remove_day4=True):
        """
        Splits dataframes into test and training, while test only contains day 3 transitions
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = []      

        for df in self.dataframes:
            header = set(df['label'].tolist())
            trails = set()
            if remove_day4:
                # Removing Day 4 Trails
                for i in header:
                    trail = eval(i)
                    if trail[0] != 4:
                        trails.add(i)
                header = trails
            for trial in header:
                # geting rows with (day, Trail)-label
                rows = df.loc[df['label'] == trial].to_numpy()
                # getting binary response label
                response = rows[0][-1]#1 if (rows[0][-1] > 0) else 0
                # getting PC-Matrix and shuffeling PC-Arrays randomly
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)
                # shuffle PC-Matrix
                np.random.shuffle(rows)

                # Spliiting in Training/Test depending on transition
                for i in range(len(rows)):
                    if eval(trial)[0] != 3:
                        X_train.append(rows[i])
                        y_train.append(response)
                    else:
                        X_test.append(rows[i])
                        y_test.append(response)

        # Now each training/test part contains data from all trails
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def population_splitter(self, test_df, path = r'D:\Dataframes\30_Transition'):
        """
        Uses self. dataframes for training-data, and the list of populations from the parameter as training
        """
        self.X_train, self.y_train = self.__split_df(self.dataframes)

        for i in range(len(test_df)):
            test_df[i] = pd.read_csv(path + '\\' + test_df[i] + '.csv')

        self.X_test, self.y_test = self.__split_df(self.dataframes)

    def __split_df2(self, dataframes, n=10, remove_day4 = True):
        """
        Takes in List of dataframes, returns X_train and y-train
        """
        X = []
        y = []
        for df in dataframes:
            header = set(df['label'].tolist())
            trails = set()
            if remove_day4:
                # Removing Day 4 Trails
                for i in header:
                    trail = eval(i)
                    if trail[0] != 4:
                        trails.add(i)
                header = trails

            for trial in header:
                # geting rows with (day, Trail)-label
                rows = df.loc[df['label'] == trial].to_numpy()
                # getting binary response label
                response = 1 if (rows[0][-1] > 0) else 0
                # getting PC-Matrix and shuffeling PC-Arrays randomly
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)

                data = []
                for i in range(n):
                    np.random.shuffle(rows)
                    for j in range(int(len(rows) / 5)):
                        a = rows[j*5: j*5+5]
                        data.append(np.concatenate(a))
                    for i in range(len(data)):
                            X.append(data[i])
                            y.append(response)

        return X, y

    def splitter_for_multiple_dataframes(self, ratio=0.75):
        """ 
        The given list of dataframes will be used to set the class attributes
        X_train, X_test, etc by taking from all trials the same ratio of data.
        """ 
        X_test = []
        X_train = []   
        y_test = []   
        y_train = []      

        for df in self.dataframes:
            header = set(df['label'].tolist())
            for trial in header:
                # geting rows with (day, Trail)-label
                rows = df.loc[df['label'] == trial].to_numpy()
                # getting binary response label
                response = rows[0][-1]#1 if (rows[0][-1] > 0) else 0
                # getting PC-Matrix and shuffeling PC-Arrays randomly
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)
                # shuffle PC-Matrix
                np.random.shuffle(rows)
                # Adding first part to training data, rest is test-data
                cut = int(ratio*len(rows))
                for i in range(len(rows)):
                    if i < cut:
                        X_train.append(rows[i])
                        y_train.append(response)
                    else:
                        X_test.append(rows[i])
                        y_test.append(response)

        # Now each training/test part contains data from all trails

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def split_data(self, n=10, ratio=0.8, remove_day4=True):
        """ 
        The given list of dataframes will be used to set the class attributes
        X_train, X_test, etc by taking from all trials the same ratio of data.
        """ 
        X_test = []
        X_train = []   
        y_test = []   
        y_train = []      

        for df in self.dataframes:
            header = set(df['label'].tolist())

            trails = set()
            if remove_day4:
                # Removing Day 4 Trails
                for i in header:
                    trail = eval(i)
                    if trail[0] != 4:
                        trails.add(i)
                header = trails

            for trial in header:
                # geting rows with (day, Trail)-label
                rows = df.loc[df['label'] == trial].to_numpy()
                # getting binary response label
                response = rows[0][-1]#1 if (rows[0][-1] > 0) else 0
                # getting PC-Matrix and shuffeling PC-Arrays randomly
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)

                data = []
                #for i in range(n):
                np.random.shuffle(rows)
                for j in range(int(len(rows) / 5)):
                    a = rows[j*5: j*5+5]
                    data.append(np.concatenate(a))

                # Adding first part to training data, rest is test-data
                cut = int(ratio*len(data))
                for i in range(len(data)):
                    if i < cut:
                        X_train.append(data[i])
                        y_train.append(response)
                    else:
                        X_test.append(data[i])
                        y_test.append(response)


        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def shuffle_labels(self):
        """
        shuffles labels to have a (random) Benchamrk
        """
        random.shuffle(self.y_train)
        random.shuffle(self.y_test)

    def use_SMOTE(self):
        """performs SMOTE on training data"""
        smote = SMOTE()
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def use_ADASYN(self):
        """performs ADASYN on training data"""
        ada = ADASYN()
        self.X_train, self.y_train = ada.fit_resample(self.X_train, self.y_train)

    def grid_search(self, C, Y = [1], kernel='rbf', degree=3, class_weight='balanced'):
        """
        Performs Grid-Search on given classifier
        """
        results = []
        for c in range(len(C)):
            results.append([])
            for y in range(len(Y)):
                svm = SVC(kernel=kernel, C=C[c], degree=degree, gamma=Y[y], class_weight=class_weight).fit(self.X_train, self.y_train)
                self.classifier = svm
                f1 = metrics.f1_score(self.y_test, self.classifier.predict(self.X_test), average="weighted")
                results[c].append(round(f1,4))
        
        ax = sns.heatmap(results, annot=True, vmin=0, vmax=1, xticklabels=C, yticklabels=Y)
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.title("Grid Search:\n RBF-Kernel, balanced class_weights")
        plt.show()    

    def do_Logistic_Regression(self, penality='l2', c=1.0):
        """
        performs logistic regression 
        """
        # Sources:
        # https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        LR = LogisticRegression(penalty=penality,  C=c).fit(
            self.X_train, self.y_train)
        self.accuracy = LR.score(self.X_test, self.y_test)
        self.classifier = LR

    def do_LR_CV(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', solver='lbfgs', class_weight=None):
        """Logistic Regression CV (aka logit, MaxEnt) classifier.
        Taken/Copied From official Doku:
        https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/linear_model/_logistic.py#L1502
        ---------------------------------------------------------
        Cs: int or list of floats, default=10
            describes the inverse of regularization strength

        fit_intercept: Bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) 
            should be added to the decision function.

        cv : int or cross-validation generator, default=None
            cross-validation generator used is Stratified K-Folds

        dual: bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for
            l2 penalty with liblinear solver. Prefer dual=False when
            n_samples > n_features.

        penalty: {'l1', 'l2', 'elasticnet'}, default='l2'
            'elasticnet' is only supported by the 'saga' solver.

        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'

        class_weight : dict or 'balanced', default=None
        """
        LRCV = LogisticRegressionCV(Cs=Cs, fit_intercept=fit_intercept, cv=cv, penalty=penalty, solver=solver, class_weight=class_weight).fit(
            self.X_train, self.y_train)
        self.accuracy = LRCV.score(self.X_test, self.y_test)
        self.classifier = LRCV

    def do_SVM(self, kernel="linear", degree=3, c=1, gamma='scale', class_weight=None):
        """
        performs Support Vectors Machine on dataset
        """
        svm = SVC(kernel=kernel, C=c, degree=degree, gamma=gamma, class_weight=class_weight).fit(self.X_train, self.y_train)
        self.accuracy = svm.score(self.X_test, self.y_test)
        self.cm = metrics.confusion_matrix(self.y_test, svm.predict(self.X_test), normalize='true')
        self.classifier = svm

    def get_cm(self):
        """ returns Confusion matrix"""
        return self.cm

    def get_accuracy(self):
        """ returns accuracy"""
        return self.accuracy

    def get_f1(self, avg='binary'):
        """ returns f1-Score"""
        return metrics.f1_score(self.y_test, self.classifier.predict(self.X_test), average=avg)

    def plot_CM(self, norm: str=None, title: str='Confusion Matrix', path_dir: str=None):
        """
        plots Confusion Matrix of results, can be saved to path
        :param norm (str) - True for normalized CM (normalize must be one of {'true', 'pred', 'all', None})
        :param title (str) - Title for the CM
        :param path_dir (str) - if provided, saves CM to that directory
        """
        # Source:
        # https://stackoverflow.com/questions/57043260/how-change-the-color-of-boxes-in-confusion-matrix-using-sklearn
        class_names = ['0->1', '0->0', '1->0', '1->1']
        disp = metrics.plot_confusion_matrix(self.classifier, self.X_test, self.y_test,
                                             display_labels=class_names,
                                             cmap=plt.cm.OrRd,
                                             normalize=norm,
                                             values_format='.3f',
                                             labels=class_names)
        
        disp.ax_.set_title(title)
        if path_dir == None:
            plt.show()
        else:
            plt.savefig(path_dir + '\\CM.png')


def test_SVM():
    a = NeuralEarthquake_Classifier(
    r"D:\Dataframes\20PCs\bl693_no_white_Pop05.csv", 'bl693_no_white_Pop05')
    a.prepare_binary_labels()
    a.do_SVM(kernel='sigmoid', degree=3, c=1)
    cm = a.get_cm()
    acc = a.get_accuracy()
    print(cm)
    print(acc)

def get_n_random(n, remove=None, path=r'D:\Dataframes\100_Transition'):
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in range(len(files)):
        files[i] = files[i][:-4]
    for i in remove:
        if i in files: files.remove(i)
    test = random.sample(files, n)
    print(test)
    return test


a = Classifier(['bl693_no_white_Pop05', 'bl693_no_white_Pop02'], r'D:\Dataframes\tSNE\perp30')
#a.add_dataframes(['bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=p)
a.split_population_wise(1, remove_day4=False)
a.print_shape()
#a.random_split()
#a.splitter_for_multiple_dataframes()
#a.split_transitions()
#a.use_SMOTE()
#a.shuffle_labels()
#a.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced') #class_weight='balanced'

#print("Macro: ",a.get_f1(avg="macro"))
#print("Micro: ", a.get_f1(avg="micro"))
#print("Weighted: ",a.get_f1(avg="weighted"))
#a.plot_CM(title='bl693_no_white_Pop05')
#c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000, 10000]
#a.grid_search(C=c, Y=c)
