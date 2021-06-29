from sklearn.svm import SVC
import os
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from get_data_for_DL import getMeanAndVariance

class NeuralEarthquake_Classifier():

    def __init__(self, path, population):
        self.dataframe = pd.read_csv(path)
        self.population = population
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self. accuracy = None
        self.cm = None
        self.dataframes = [self.dataframe]

    def read_in_df(self, path):
        self.dataframe = pd.read_csv(path)

    def random_split(self, split_ratio=0.2, randomState=None, strat=None):
        """
        Converts dataframe into list with PC's and labels,
        saves X_train, y_train, X_test, y_test as class attributes and sets dataframe to None after.
        Optional variable split_ratio sets ratio for training/test split, bist be between 0 and 1.
        randomState must be int, for reproducable outcomes
        """
        X = []
        y = []

        for df in self.dataframes:
            for index, row in df.iterrows():
                if eval(row[1])[0] != 4:
                    X.append(row[2:-1].tolist())
                    y.append(row[-1])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=randomState, stratify=strat)
        X, y = None, None

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
                response = 1 if (rows[0][-1] > 0) else 0
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

    def __split_df(self, dataframes, n=10, remove_day4 = True):
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

    def splitter_for_multiple_dataframes(self, ratio=0.8):
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
                response = 1 if (rows[0][-1] > 0) else 0
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
        shuffles y-labels to have a f1-Score Benchamrk
        """
        random.shuffle(self.y_train)
        random.shuffle(self.y_test)

    def use_SMOTE(self):
        """performs SMOTE on training data"""
        smote = SMOTE()
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        #self.X_test, self.y_test = smote.fit_resample(self.X_test, self.y_test)

    def use_ADASYN(self):
        """performs ADASYN on training data"""
        ada = ADASYN()
        self.X_train, self.y_train = ada.fit_resample(self.X_train, self.y_train)
        #self.X_test, self.y_test = ada.fit_resample(self.X_test, self.y_test)

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
                f1 = metrics.f1_score(self.y_test, self.classifier.predict(self.X_test))
                results[c].append(round(f1,4))
        
        ax = sns.heatmap(results, annot=True, vmin=0, vmax=1, xticklabels=C, yticklabels=Y)
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.title("Grid Search:\n RBF-Kernel, balanced class_weights")
        plt.show()

        #clf = GridSearchCV(classifier, parameters, scoring = score, cv = cv)
        #clf.fit(self.X_train, self.y_train)

        #print(clf.best_estimator_)
        #print(clf.score(self.X_test, self.y_test))
        #df = pd.DataFrame.from_dict(clf.cv_results_)
        #df.to_csv(r'C:\Users\Sam\Desktop\grid1Pop.csv')
           

    def get_accuracy(self):
        return self.accuracy

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

    def do_SVM(self, kernel="linear",degree=3, c=1, gamma='scale', class_weight=None):
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

    def plot_CM(self, norm=None, title=None, path=None):
        """
        plots Confusion Matrix of results, can be saved to path
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
        title = "Confusion Matrix of {},:\n {}'bl693_no_white_Pop02', 'bl693_no_white_Pop03'\n with {} Dimensions, total accuracy: {}".format(
            str(self.classifier), self.population, len(self.X_train[0]), str(round(self.accuracy, 4)))
        disp.ax_.set_title(title)
        if path == None:
            plt.show()
        else:
            plt.savefig(path)

    def merge_dataframes(self, lst, path = r"D:\Dataframes\20PCs"):
        """Takes in list of other populations, adds them to dataframe"""
        #removing index column
        # Source: https://stackoverflow.com/questions/43983622/remove-unnamed-columns-in-pandas-dataframe
        self.dataframe = self.dataframe.drop(self.dataframe.columns[self.dataframe.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        dfs = []
        for i in lst:
            # Reading new dataframes and removing index column
            df = pd.read_csv(path + '\\' + i + '.csv')
            df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            dfs.append(df)
        # merging dataframes
        self.dataframe = pd.concat([self.dataframe] + dfs, axis=0)

    def add_dataframes(self, lst, path = r"D:\Dataframes\20PCs"):
        """Takes in list of other populations, adds them to dataframe list"""
        for i in lst:
            # Reading new dataframes and removing index column
            self.dataframes.append(pd.read_csv(path + '\\' + i + '.csv'))



def test_SVM():
    a = NeuralEarthquake_Classifier(
    r"D:\Dataframes\20PCs\bl693_no_white_Pop05.csv", 'bl693_no_white_Pop05')
    a.prepare_binary_labels()
    a.do_SVM(kernel='sigmoid', degree=3, c=1)
    cm = a.get_cm()
    acc = a.get_accuracy()
    print(cm)
    print(acc)

def get_n_random(n, remove=None, path=r'D:\Dataframes\30_Transition'):
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in range(len(files)):
        files[i] = files[i][:-4]
    for i in remove:
        if i in files: files.remove(i)
    test = random.sample(files, n)
    print(test)
    return test

p = r'D:\Dataframes\isomap_multi_2d'
#p = r'D:\Dataframes\30_most_active'
a = NeuralEarthquake_Classifier(p + '\\' + 'bl693_no_white_Pop05.csv', 'bl693_no_white_Pop05')
#a.add_dataframes(['bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=p)
a.random_split()
#a.splitter_for_multiple_dataframes()
#a.split_transitions()
#a.population_splitter(['bl684_no_white_Pop03', 'bl689-1_one_white_Pop09', 'bl688-1_one_white_Pop05', 'bl709_one_white_Pop11', 'bl660-1_two_white_Pop07'])
#a.split_data()
#a.use_SMOTE()
#a.use_ADASYN()
#a.shuffle_labels()
#a.prepare_binary_labels()
a.do_SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced') #class_weight='balanced'
#print(a.get_f1())
print("Macro: ",a.get_f1(avg="macro"))
print("Micro: ", a.get_f1(avg="micro"))
print("Weighted: ",a.get_f1(avg="weighted"))
#print("Accuracy: ",a.get_accuracy())
a.plot_CM()
#c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000, 10000]
#a.grid_search(C=c, Y=c)
