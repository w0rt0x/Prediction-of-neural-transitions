from imblearn.over_sampling._smote.base import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')
from get_data_for_DL import get_data, use_adasyn, use_smote, encode_labels, decode_labels
from keras import backend as K
import tensorflow as tf

def f1_weighted(true, pred): #shapes (batch, 4)
    # Source for weightend f1:
    # https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 4) 


    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
        #all with shape (4,)
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'

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


X_train, X_test, y_train, y_test = get_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\30_Transition_multiclass')
X_train, X_test, y_train, y_test = use_smote(X_train, X_test, y_train, y_test)
#X_train, X_test, y_train, y_test = use_adasyn(X_train, X_test, y_train, y_test)
# Preparing numerical labels, Keras does not allow strings!
y_train = encode_labels(y_train)
y_test_en = encode_labels(y_test)

dim = len(X_train[0])

# Starting Keras Model
# Tutorial:
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
print('starting model...')

# Sequentiel model, layers are added one after another
dl = Sequential()
# input_dim is number of input variables
# Dense class is fully connected layer
# first number defines number of neurons
# activation-function is set to relu
# final layer has sigmoid so that the result is in [0,1]
dl.add(Dense(75, input_dim=dim, activation='sigmoid')) # 'tanh', oder 'sigmoid' oder relu
dl.add(Dense(38, activation='sigmoid'))
dl.add(Dense(19, activation='sigmoid'))
dl.add(Dense(10, activation='sigmoid'))
dl.add(Dense(4,activation='softmax')) #in case of binary: Sigmoid and just one output

# most confusing thing:
# Input size is given to first hidden layer!

print('model initialized!')
print('Compiling model ...')

# Choosing the loss-function, more infos here:
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# Also choosing optimizer (stochastic gradient descent algorithm 'adam'):
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# Using accuracy-metric because of binary classification
dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])

print('Model compiled!')
print('Fitting model...')
# Fitting is done with Epochs, each epoch contains batches:
# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# batch size is a number of samples processed before the model is updated
# number of epochs is the number of complete passes through the training dataset
dl.fit(X_train, y_train, epochs=100, batch_size=5)
print('Model fitted!')

print('Evaluating model:')
_ , accuracy = dl.evaluate(X_test, y_test_en)
print('Accuracy %.2f' % (accuracy*100))

print(' ')
y_pred = decode_labels(dl.predict_classes(X_test))
print("F1-Score wighted: ", metrics.f1_score(y_test, y_pred,average='weighted'))
print("F1-Score macro: ", metrics.f1_score(y_test, y_pred,average='macro'))
print("F1-Score micro: ", metrics.f1_score(y_test, y_pred,average='micro'))
print(classification_report(y_test, y_pred))

# Evaluation
# https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
# Advanced Tut:
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# Saved Model to file:
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
