from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
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
        self.cm = None

    def read_in_df(self, path):
        self.dataframe = pd.read_csv(path)

    def prepare_binary_labels(self, split_ratio=0.2, randomState=None, strat=None):
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
            # Binary Labels for activity
            if row[-1] > 0:
                y.append(1)
            else:
                y.append(0)

        self.dataframe = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=randomState, stratify=strat)
        X, y = None, None

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

    def do_SVM(self, kernel="linear",degree=3, c=1, gamma='scale'):
        """
        performs Support Vectors Machine on dataset
        """
        svm = SVC(kernel=kernel, C=c, degree=degree, gamma=gamma).fit(self.X_train, self.y_train)
        self.accuracy = svm.score(self.X_test, self.y_test)
        self.cm = metrics.confusion_matrix(self.y_test, svm.predict(self.X_test), normalize='true')
        self.classifier = svm

    def get_cm(self):
        """ returns Confusion matrix"""
        return self.cm

    def get_accuracy(self):
        """ returns accuracy"""
        return self.accuracy

    def get_f1(self):
        """ returns f1-Score"""
        return metrics.f1_score(self.y_test, self.classifier.predict(self.X_test))

    def plot_CM(self, norm=None, title=None, path=None):
        """
        plots Confusion Matrix of results, can be saved to path
        """
        # Source:
        # https://stackoverflow.com/questions/57043260/how-change-the-color-of-boxes-in-confusion-matrix-using-sklearn
        class_names = set(self.y_train)
        disp = metrics.plot_confusion_matrix(self.classifier, self.X_test, self.y_test,
                                             display_labels=class_names,
                                             cmap=plt.cm.OrRd,
                                             normalize=norm,
                                             values_format='.3f')
        title = "Confusion Matrix of {}:\n {} with {} Dimensions, total accuracy: {}".format(
            str(self.classifier), self.population, len(self.X_train[0]), str(round(self.accuracy, 4)))
        disp.ax_.set_title(title)
        if path == None:
            plt.show()
        else:
            plt.savefig(path)

def test_SVM():
    a = NeuralEarthquake_Classifier(
    r"D:\Dataframes\20PCs\bl693_no_white_Pop05.csv", 'bl693_no_white_Pop05')
    a.prepare_binary_labels()
    a.do_SVM(kernel='sigmoid', degree=3, c=1)
    cm = a.get_cm()
    acc = a.get_accuracy()
    print(cm)
    print(acc)


#test_SVM()
# bl693_no_white_Pop06
a = NeuralEarthquake_Classifier(
    r"D:\Dataframes\20PCs\bl693_no_white_Pop05.csv", 'bl693_no_white_Pop05')
a.prepare_binary_labels()
a.do_LR_CV(Cs=5, fit_intercept=False, cv=10)
print(a.get_f1())
a.plot_CM()
a.do_SVM(kernel='rbf', c=1, gamma=100)
a.plot_CM()
print(a.get_f1())
