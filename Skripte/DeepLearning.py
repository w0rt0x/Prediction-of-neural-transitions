from getter_for_populations import sort_out_populations
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class FeedforwardNetWork():
    
    def __init__(self, input_dim: int=40, hidden_dim: int=16, loss: str='categorical_crossentropy', optimizer: str='adam', metric: str='accuracy', epochs: int=50, batch_size: int=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size

    def set_data(self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array):
        """
        Sets Training and test-data, must be numpy arrays
        """
        self.X_train = X_train.tolist()
        self.X_test = X_test.tolist()
        self.y_train = y_train.tolist()
        self.y_test = y_test.tolist()

        self.y_train = self.__encode_labels(self.y_train)
        self.y_test_en = self.__encode_labels(self.y_test)

    def train(self):
        """
        Trains model
        """
        self.makeModell()
        self.fitModel()

    def makeModell(self):
        # Sequentiel model, layers are added one after another
        dl = Sequential()
        dl.add(Dense(self.hidden_dim, input_dim=self.input_dim, activation='sigmoid'))
        dl.add(Dense(4,activation='softmax'))

        # Choosing the loss-function, more infos here:
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        # Also choosing optimizer (stochastic gradient descent algorithm 'adam'):
        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # Using accuracy-metric because of binary classification
        dl.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
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

    def fitModel(self, verbose: int=1):
        """
        Fits Model with given Epoch and Batch Size,
        :param verbose(int=[1,2,3]) - displays fitting process, 0 is no display at all
        """
        # Fitting is done with Epochs, each epoch contains batches:
        # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
        # batch size is a number of samples processed before the model is updated
        # number of epochs is the number of complete passes through the training dataset
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

    def predict(self, return_f1s: bool=True):
        """
        performs Prediction on dataset
        :param return_f1s (bool, default is True) - If True: returns micro, macro and weighted f1-Score
        """
        self.pred = self.__decode_labels(self.model.predict_classes(self.X_test))
        self.report = classification_report(self.y_test, self.pred, output_dict=True)

        if return_f1s:
            return self.report['accuracy'], self.report['macro avg']['f1-score'], self.report['weighted avg']['f1-score']

    def get_CM(self, order: list=['0->0', '0->1', '1->0', '1->1']) -> np.array:
        """
        returns confusion matrix
        """
        return confusion_matrix(self.y_test, self.pred, labels=order)

    def get_report(self) -> dict:
        """
        returns scikit classification report as dictionary
        """
        return self.report

    def plotWeights(self):
        """
        Plots first Layer Weights as heatmap
        """
        weights = self.model.layers[0].get_weights()[0]
        plt.imshow(weights, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.ylabel('Input-Layer')
        plt.xlabel('First Layer')
        plt.title('First Layer Weights')
        plt.show()

    def map_input(self, title: str, show: bool=True, save: str=None):
        """
        Maps mean of all correct predicted classes onto the first Layer weights,
        showing 4 heatmaps (1 per class) in total
        """
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
        fig.colorbar(im, cax=axis[-1], shrink=0.5)
        plt.tight_layout()

        if show:
            plt.show()
        if save != None:
            plt.savefig(save)

        plt.clf()
        plt.cla()
        plt.close()
