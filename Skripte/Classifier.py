from sklearn.svm import SVC
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
from sklearn.metrics import confusion_matrix



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

        for df in self.dataframes:
            X, x, Y, y = self.__split_df(df, 1 - split_ratio, remove_day4, shuffle)
            X_train = X_train + X
            y_train = y_train + Y
            X_test = X_test + x
            y_test = y_test + y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

    def split_day_wise(self, day:int=3, remove_day4:bool=True, shuffle: bool=True):
        """
        Splits each Populations into training/test data, while day x is only used for testing
        :param day (int) - default is day3 that is used for testing
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
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
                # getting response label
                response = rows[0][-1]
                # getting the actual data from the matrix
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)
                for i in range(len(rows)):
                    if eval(trial)[0] == day:
                        X_test.append(rows[i])
                        y_test.append(response)
                    else:
                        X_train.append(rows[i])
                        y_train.append(response)

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)  

    def split_trial_wise_with_concat_vectors(self, n_vec: int, split_ratio: float=0.2, remove_day4: bool=True, shuffle: bool=True):
        """
        Each Population has ~20 repetitions per trial. This function splits each of those repetitions so that 
        training and test data have some repetitions (split_ratio).
        variable number of repetitions will be concatinated.
        :param n_vec (int) - number of repetiotions that are concatinated to one trial
        :param Split-ratio (float) - Ratio of Training/Test Split, default is 0.2
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

        for df in self.dataframes:
            X, x, Y, y = self.__split_df(df, 1 - split_ratio, remove_day4, shuffle, n_vec=n_vec)
            X_train = X_train + X
            y_train = y_train + Y
            X_test = X_test + x
            y_test = y_test + y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)       

    def __split_df(self, df:pd.DataFrame, ratio:float, rem_day4:bool, shuffle:bool, n_vec: int=1) -> Tuple[list, list, list, list]:
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
            if trail[0] != 4:
                trails.add(i)
            else:
                if not(rem_day4):
                    trails.add(i)

        header = trails

        # Getting all the matrices from the trials
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

            if n_vec == 1:
                pass
            else:
                new_rows = []
                # taking samples
                while len(rows) > n_vec:
                    vecs = rows[:n_vec]
                    # deleting vectors that are already taken
                    rows = rows[n_vec:]
                    # Concat vectors to one
                    new_rows.append(np.concatenate(vecs))
                rows = new_rows

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

    def grid_search(self, title, C, Y = [1], kernel='rbf', degree=3, class_weight='balanced', show: bool=True, dest_path: str = None):
        """
        Performs Grid-Search on given classifier
        """
        results = []
        for c in range(len(C)):
            print(C[c])
            results.append([])
            for y in range(len(degree)):
                #svm = SVC(kernel=kernel, C=C[c], degree=degree, gamma=Y[y], class_weight=class_weight).fit(self.X_train, self.y_train)
                svm = SVC(kernel=kernel, C=C[c], degree=degree[y], class_weight=class_weight).fit(self.X_train, self.y_train)
                self.classifier = svm
                f1 = metrics.f1_score(self.y_test, self.classifier.predict(self.X_test), average="weighted")
                results[c].append(round(f1,4))
        #ax = sns.heatmap(results, annot=True, vmin=0, vmax=1, xticklabels=Y, yticklabels=C, cbar_kws={'label': 'weighted f1-Score'})
        ax = sns.heatmap(results, annot=True, vmin=0, vmax=1, xticklabels=degree, yticklabels=C, cbar_kws={'label': 'weighted f1-Score'})
        plt.xlabel('Degree')
        plt.ylabel('C')
        plt.title(title)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path)

        plt.clf()
        plt.cla()
        plt.close()   

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

    def do_SVM(self, kernel="linear", degree=3, c=1, gamma='scale', class_weight=None, print_report: bool=False):
        """
        performs Support Vectors Machine on dataset
        """
        svm = SVC(kernel=kernel, C=c, degree=degree, gamma=gamma, class_weight=class_weight).fit(self.X_train, self.y_train)
        self.classifier = svm
        # Classification report as dictionary
        self.pred = svm.predict(self.X_test)
        self.report = classification_report(self.y_test, self.pred, output_dict=True)
        if print_report:
            print(classification_report(self.y_test, svm.predict(self.X_test)))

    def get_report(self):
        """
        returns scikit classification report as dictionary
        """
        return classification_report(self.y_test, self.pred, output_dict=True)

    def get_predictions(self):
        """
        returns predicted labels
        """
        return self.pred

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
        class_names = ['0->0', '0->1', '1->0', '1->1']
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

    def get_data(self):
        """
        returns X_train, X_test, etc
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_cm(self) -> np.array:
        """
        returns confusion matrix
        """
        return confusion_matrix(self.y_test, self.pred, labels=['0->0', '0->1', '1->0', '1->1'])
