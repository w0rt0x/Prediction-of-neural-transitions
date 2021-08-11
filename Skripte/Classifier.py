from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class SVMclassifier():

    def __init__(self,  kernel: str="rbf", degree: int=3, c: float=1.0, gamma: float=0.5, class_weight: str="balanced"):
        """
        Init for SVM Classifier
        :param kernel (str) - can be "poly", "linear" or "rbf"(default)
        :param degree (int) - only relevent for "poly" Kernel, default is 3
        :param c (float) - default is 1.0
        :param gamma (float) - only relevant for "rbf" Kernel, default is 0.5       
        :param class_weight - default is balanced to to imbalanced data sets
        """
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight

    def set_data(self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array):
        """
        Sets Training and test-data, must be numpy arrays
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def grid_search(self, title: str, C: list, Y: list=[1], show: bool=True, dest_path: str = None):
        """
        Performs Grid-Search on SVM
        :param title (str) - Title of Plot
        :param C (list) - list of c-values that should be tested
        :param Y (list) - list of gamma-values that should be tested, default is [1]
        :param show (bool, default True) - if True: shows Plot
        :param dest_path (str, default is None) - if provided, plot will be saved to path. path must include name and format of plot
        """
        results = []
        for c in range(len(C)):
            print(C[c])
            results.append([])
            for y in range(len(Y)):
                svm = SVC(kernel=self.kernel, C=C[c], degree=self.degree, gamma=Y[y], class_weight=self.class_weight).fit(self.X_train, self.y_train)
                self.classifier = svm
                f1 = metrics.f1_score(self.y_test, self.classifier.predict(self.X_test), average="weighted")
                results[c].append(round(f1,4))
        
        ax = sns.heatmap(results, annot=True, vmin=0, vmax=1, xticklabels=Y, yticklabels=C, cbar_kws={'label': 'weighted f1-Score'})
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.title(title)

        if show:
            plt.show()

        if dest_path !=None:
            plt.savefig(dest_path)

        plt.clf()
        plt.cla()
        plt.close()   

    def predict(self, return_f1s: bool=True):
        """
        performs Support Vectors Machine on dataset
        :param return_f1s (bool, default is True) - If True returns micro, macro and weighted f1-Score
        """
        svm = SVC(kernel=self.kernel, C=self.c, degree=self.degree, gamma=self.gamma, class_weight=self.class_weight, cache_size=2000).fit(self.X_train, self.y_train)
        self.classifier = svm
        # Classification report as dictionary
        self.pred = svm.predict(self.X_test)
        self.report = classification_report(self.y_test, self.pred, output_dict=True)

        if return_f1s:
            return self.report['accuracy'], self.report['macro avg']['f1-score'], self.report['weighted avg']['f1-score']

    def get_report(self) -> dict:
        """
        returns scikit classification report as dictionary
        """
        return self.report

    def get_predictions(self)->np.array:
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

    def get_CM(self, order: list=['0->0', '0->1', '1->0', '1->1']) -> np.array:
        """
        returns confusion matrix
        """
        return confusion_matrix(self.y_test, self.pred, labels=order)


svm = SVMclassifier()
from data_holder import Data
d = Data(['bl693_no_white_Pop06'], r'D:\Dataframes\most_active_neurons\40')
d.split_trial_wise()
d.use_SMOTE()
X, x, Y, y = d.get_data()
svm.set_data(X, x, Y, y)
print(svm.predict())
print(svm.plot_CM())