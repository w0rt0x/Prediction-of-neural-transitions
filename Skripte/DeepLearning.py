import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from scipy.io import loadmat
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def get_PCA_data(pop):
    path = r"D:\Dataframes\20PCs\{}.csv".format(pop)
    dataframe = pd.read_csv(path)
    X = []
    y = []

    for index, row in dataframe.iterrows():
        X.append(row[2:-1].tolist())
        # Binary Labels for activity
        if row[-1] > 0:
            y.append(1)
        else:
            y.append(0)

    return train_test_split(X, y, test_size=0.2)

X_train, X_test, y_train, y_test = get_PCA_data('bl693_no_white_Pop06')

dim = 20

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
dl.add(Dense(12, input_dim=dim, activation='relu'))
dl.add(Dense(12, activation='relu'))
dl.add(Dense(12, activation='relu'))
dl.add(Dense(12, activation='relu'))
dl.add(Dense(1,activation='sigmoid'))

# most confusing thing:
# Input size is given to first hidden layer!

print('model initialized!')
print('Compiling model ...')

# Choosing the loss-function, more infos here:
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# Also choosing optimizer (stochastic gradient descent algorithm 'adam'):
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# Using accuracy-metric because of binary classification
dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Model compiled!')
print('Fitting model...')
# Fitting is done with Epochs, each epoch contains batches:
# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# batch size is a number of samples processed before the model is updated
# number of epochs is the number of complete passes through the training dataset
dl.fit(X_train, y_train, epochs=100, batch_size=5)
print('Model fitted!')

print('Evaluating model:')
_ , accuracy = dl.evaluate(X_test, y_test)
print('Accuracy %.2f' % (accuracy*100))

print(' ')
y_pred = dl.predict_classes(X_test)
print(classification_report(y_test, y_pred))

# Evaluation
# https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
# Advanced Tut:
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# Saved Model to file:
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
