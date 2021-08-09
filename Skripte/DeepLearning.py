from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from Classifier import Classifier
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Plotter import Plotter, get_all_pop
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple


class FeedforwardNetWork():
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_test_en = None

    def get_data(self, liste=['bl693_no_white_Pop05', 'bl693_no_white_Pop01', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03', 'bl693_no_white_Pop04', 'bl693_no_white_Pop06'], path=r'D:\Dataframes\most_active_neurons\40', shuffle: bool=False, smote: bool=True):
        """
        sets training and tes data, takes in list with filenames from directory and path that that directory
        """
        a = Classifier(liste, path)
        a.split_trial_wise()
        if shuffle:
            a.shuffle_labels()
        if smote:
            a.use_SMOTE()
        self.X_train = a.X_train.tolist()
        self.X_test = a.X_test.tolist()
        self.y_train = a.y_train.tolist()
        self.y_test = a.y_test.tolist()

    def makeModell(self, loss='categorical_crossentropy', optimizer='adam', metric='accuracy'):
        # Sequentiel model, layers are added one after another
        dim = len(self.X_train[0])
        dl = Sequential()
        dl.add(Dense(16, input_dim=len(self.X_train[0]), activation='sigmoid'))
        dl.add(Dense(4,activation='softmax'))

        # Choosing the loss-function, more infos here:
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        # Also choosing optimizer (stochastic gradient descent algorithm 'adam'):
        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # Using accuracy-metric because of binary classification
        dl.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        self.model = dl

    def __encode_labels(self, y):
        """ Encodes string labels"""
        y_hot = []
        table = {"0->0":[1,0,0,0], "0->1":[0,1,0,0], "1->0":[0,0,1,0], "1->1":[0,0,0,1]}
        for i in range(len(y)):
            y_hot.append(table[y[i]])
        return y_hot

    def __decode_labels(self, y):
        """ Encodes string labels"""
        y_hot = []
        table = {0:"0->0", 1:"0->1", 2:"1->0", 3:"1->1"}
        for i in range(len(y)):
            y_hot.append(table[y[i]])
        return y_hot
    
    def encode_labels(self):
        """
        encodes labels, One Hot Encoding
        """
        self.y_train = self.__encode_labels(self.y_train)
        self.y_test_en = self.__encode_labels(self.y_test)

    def fitModel(self, epochs=50, batch_size=32):
        """
        Fits Model with given Epoch and Batch Size, default: epochs=50, batch_size=32
        """
        # Fitting is done with Epochs, each epoch contains batches:
        # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
        # batch size is a number of samples processed before the model is updated
        # number of epochs is the number of complete passes through the training dataset
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def evaluateModel(self, show_report: bool=True):
        """
        Prints Classification report
        returns CM
        """
        _ , accuracy = self.model.evaluate(self.X_test, self.y_test_en)
        if show_report:
            print('Accuracy %.2f' % (accuracy*100))

            print(' ')
        y_pred = self.__decode_labels(self.model.predict_classes(self.X_test))
        if show_report:
            print("F1-Score macro: ", metrics.f1_score(self.y_test, y_pred,average='macro'))
            print("F1-Score micro: ", metrics.f1_score(self.y_test, y_pred,average='micro'))
            print("F1-Score wighted: ", metrics.f1_score(self.y_test, y_pred,average='weighted'))
            print(classification_report(self.y_test, y_pred))
        return confusion_matrix(self.y_test, y_pred, labels=['0->0', '0->1', '1->0', '1->1'])

    def get_f1_scores(self) -> Tuple[float, float, float]:
        """
        returnsmacro, micro and weighted f1 scores
        """
        y_pred = self.__decode_labels(self.model.predict_classes(self.X_test))
        return metrics.f1_score(self.y_test, y_pred,average='macro'), metrics.f1_score(self.y_test, y_pred,average='micro'), metrics.f1_score(self.y_test, y_pred,average='weighted')

    def mapMeanWeights(self, layer=0):
        """
        Maps Wights to heatmap
        """
        weights = self.model.layers[0].get_weights()[0]
        means = np.zeros(len(weights))
        for i in range(len(weights)):
            means[i] = np.mean(weights[i])
        weights =  np.array(np.array_split(means, 5))
        plt.imshow(weights, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.ylabel('Trials')
        plt.xlabel('Neuronen')
        plt.title('Mean of first-layer weights')
        plt.show()

    def plotWeights(self, layer=0):
        """
        Plots Layer Weights
        """
        weights = self.model.layers[0].get_weights()[0]
        plt.imshow(weights, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.ylabel('Input-Layer')
        plt.xlabel('First Layer')
        plt.title('First Layer Weights')
        plt.show()

    def map_input(self, title: str, show: bool=True, save: str=None):
        weights = self.model.layers[0].get_weights()[0]
        weights = np.asarray(weights)
        y_pred = self.__decode_labels(self.model.predict_classes(self.X_test))
        d = dict()

        for i in range(len(y_pred)):
            # Only for correct predictions
            if y_pred[i] == self.y_test[i]:
                if y_pred[i] in d:
                    d[y_pred[i]].append(self.X_test[i])
                else:
                    d[y_pred[i]] = [self.X_test[i]]
        

        
        keys = ['0->0', '0->1', '1->0', '1->1']
        samples = []
        for key in keys:
            samples.append(np.mean(d[key], axis=0))
        del d

        matrices = []
        for i in range(len(samples)):
            w = deepcopy(weights)
            vec = samples[i]
            for j in range(len(vec)):
                w[j] = weights[j] * vec[j]

            matrices.append(w)

        matrices = np.asarray(matrices)
        mini = np.min(matrices)
        maxi = np.max(matrices)
        
        fig, axis = plt.subplots(nrows=1, ncols=5, gridspec_kw={'width_ratios': [4, 4, 4, 4, 1]})
        
        for w in range(len(matrices)):
            im = axis[w].imshow(matrices[w], cmap='hot', interpolation='nearest', vmin=mini, vmax=maxi)
            axis[w].set_title(keys[w])
            axis[w].set_yticklabels([41, 40, 35, 30, 25, 20, 15, 10, 5])

        
        fig.suptitle(title)
        fig.supxlabel('First Hidden\n Layer')
        fig.supylabel('Input-Layer')
        
        #fig.colorbar(im,  ax=axis.ravel().tolist(), location='top', shrink=0.5)
        fig.colorbar(im, cax=axis[-1], shrink=0.5)
        plt.tight_layout()
        if show:
            plt.show()
        if save != None:
            plt.savefig(save)

        plt.clf()
        plt.cla()
        plt.close()


#populations = get_all_pop()
#a = Plotter(populations, r'D:\Dataframes\most_active_neurons\40')
#ok, nt_ok = a.sort_out_populations(show=False)

"""
CM = np.zeros((4, 4))
for pop in ok:
    a = FeedforwardNetWork()
    a.get_data(liste=[pop])
    a.encode_labels()
    a.makeModell()
    a.fitModel()
    CM = CM + a.evaluateModel(show_report=False)
    
CM_norm = np.zeros((4, 4))
for row in range(len(CM)):
    for col in range(len(CM[row])):
        CM_norm[row][col] = round(CM[row][col] / np.sum(CM[row]), 3)
CM = CM_norm

df_cm = pd.DataFrame(CM_norm, index = [i for i in ['0->0', '0->1', '1->0', '1->1']],
columns = [i for i in ['0->0', '0->1', '1->0', '1->1']])
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.title("Class-wise Normalized Confusion Matrix of all Populations with all 4 classes.\n Classification via Feedforward Network, SMOTE used on training-data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
    #path = 'C:\\Users\\Sam\Desktop\\BachelorInfo\\Bachelor-Info\\Bachelor-ML\\Skripte\\Plots\\Feedforward Mappings\\' + pop
    #a.map_input("Population: {} \n neurons-wise mean of all correct predicted trials per class\n mapped to the first Layer weights".format(pop), show=False, save=path)
"""