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

def f1_weighted(true, pred): 
    # Source for weightend f1:
    # https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 4) 

    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
        
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1 

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

    def f1_m(true, pred):
        # Source:
        # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
        y_pred = K.round(pred)
        tp = K.sum(K.cast(true*y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

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

    def evaluateModel(self):
        """
        Prints Classification report
        """
        _ , accuracy = self.model.evaluate(self.X_test, self.y_test_en)
        print('Accuracy %.2f' % (accuracy*100))

        print(' ')
        y_pred = self.__decode_labels(self.model.predict_classes(self.X_test))
        print("F1-Score macro: ", metrics.f1_score(self.y_test, y_pred,average='macro'))
        print("F1-Score micro: ", metrics.f1_score(self.y_test, y_pred,average='micro'))
        print("F1-Score wighted: ", metrics.f1_score(self.y_test, y_pred,average='weighted'))
        print(classification_report(self.y_test, y_pred))

        # Evaluation
        # https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
        # Advanced Tut:
        # https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
        # Saved Model to file:
        # https://machinelearningmastery.com/save-load-keras-deep-learning-models/


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

    def map_input(self):
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
        

        
        keys = list(d.keys())
        samples = []
        for key in keys:
            samples.append(random.choice(d[key]))
        del d

        matrices = []
        for i in range(len(samples)):
            w = deepcopy(weights)
            vec = samples[i]
            label = keys[i]
            for j in range(len(vec)):
                w[j] = weights[j] * vec[j]

            matrices.append(w)
        matrices = np.asarray(matrices)
        mini = np.min(matrices)
        maxi = np.max(matrices)
        
        for w in range(len(matrices)):
            plt.imshow(matrices[w], cmap='hot', interpolation='nearest', vmin=mini, vmax=maxi)
            plt.colorbar()
            plt.ylabel('Input-Layer')
            plt.xlabel('First Hidden\n Layer')
            plt.title(keys[w])
            plt.show()

a = FeedforwardNetWork()
a.get_data(liste=['bl693_no_white_Pop02','bl693_no_white_Pop05', 'bl693_no_white_Pop03'])
a.encode_labels()
a.makeModell()
a.fitModel()
a.evaluateModel()
a.map_input()