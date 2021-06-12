from imblearn.over_sampling._smote.base import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from get_data_for_DL import get_data, use_adasyn, use_smote, encode_labels, decode_labels
from keras import backend as K
import tensorflow as tf
import shap

class FeedforwardNetWork():
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_test_en = None

    def get_data(self, liste=['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\100_Transition_multiclass'):
        """
        sets training and tes data, takes in list with filenames from directory and path that that directory
        """
        self.X_train, self.X_test, self.y_train, self.y_test = get_data(liste, path)

    def use_smote(self):
        """
        Uses smoteon dataset
        """
        self.X_train, self.X_test, self.y_train, self.y_test = use_smote(self.X_train, self.X_test, self.y_train, self.y_test)

    def use_adasyn(self):
        """
        Uses smoteon dataset
        """
        self.X_train, self.X_test, self.y_train, self.y_test = use_adasyn(self.X_train, self.X_test, self.y_train, self.y_test)

    def makeModell(self, loss='categorical_crossentropy', optimizer='adam', metric='accuracy'):
        # Sequentiel model, layers are added one after another
        dim = len(self.X_train[0])
        dl = Sequential()
        dl.add(Dense(250, input_dim=dim, activation='sigmoid'))
        dl.add(Dense(125, activation='sigmoid'))
        dl.add(Dense(62, activation='sigmoid'))
        dl.add(Dense(31, activation='sigmoid'))
        dl.add(Dense(15, activation='sigmoid'))
        dl.add(Dense(7, activation='sigmoid'))
        dl.add(Dense(4,activation='softmax'))

        # Choosing the loss-function, more infos here:
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        # Also choosing optimizer (stochastic gradient descent algorithm 'adam'):
        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # Using accuracy-metric because of binary classification
        dl.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        self.model = dl


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

    def encode_labels(self):
        """
        encodes labels, One Hot Encoding
        """
        self.y_train = encode_labels(self.y_train)
        self.y_test_en = encode_labels(self.y_test)

    def fitModel(self, epochs=100, batch_size=5):
        """
        Fits Model with given Epoch and Batch Size, default: epochs=100, batch_size=5
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
        y_pred = decode_labels(self.model.predict_classes(self.X_test))
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

    def featureSelection(self):
        # https://github.com/slundberg/shap#deep-learning-example-with-deepexplainer-tensorflowkeras-models
        # https://eli5.readthedocs.io/en/latest/overview.html
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)

        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        shap.force_plot(explainer.expected_value, shap_values[0,:], self.X_train.iloc[0,:])
        shap.summary_plot(shap_values, self.X_train, plot_type="bar") 

a = FeedforwardNetWork()
a.get_data()
a.use_smote()
a.encode_labels()
a.makeModell()
a.fitModel()
#a.evaluateModel()
a.featureSelection()