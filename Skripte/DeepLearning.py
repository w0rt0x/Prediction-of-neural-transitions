import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
warnings.filterwarnings('ignore')

def get_data(pops, path=r'r"D:\Dataframes\20PCs', ratio=0.8, n=10):
    X_test = []
    X_train = []   
    y_test = []   
    y_train = []      

    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
        header = set(df['label'].tolist())
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

            # Adding first part to training data, rest is test-data
            cut = int(ratio*len(data))
            for i in range(len(data)):
                if i < cut:
                    X_train.append(data[i].tolist())
                    y_train.append(response)
                else:
                    X_test.append(data[i].tolist())
                    y_test.append(response)

    return X_train, X_test, y_train, y_test

def get_PCA_data(pops, path=r'r"D:\Dataframes\20PCs', ratio=0.8):
    X_test = []
    X_train = []   
    y_test = []   
    y_train = []      

    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
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
                    X_train.append(rows[i].tolist())
                    y_train.append(response)
                else:
                    X_test.append(rows[i].tolist())
                    y_test.append(response)

    return X_train, X_test, y_train, y_test
    
def use_smote(X_train, X_test, y_train, y_test):
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_test, y_test = smote.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test

def use_adasyn(X_train, X_test, y_train, y_test):
    ada = ADASYN()
    X_train, y_train = ada.fit_resample(X_train, y_train)
    X_test, y_test = ada.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data(['bl693_no_white_Pop05'], path=r'D:\Dataframes\30_Transition')
X_train, X_test, y_train, y_test = use_smote(X_train, X_test, y_train, y_test)
#X_train, X_test, y_train, y_test = use_adasyn(X_train, X_test, y_train, y_test)
dim = 150
print(len(X_train), len(X_train[0]))

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
dl.add(Dense(8, activation='relu'))
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
print("F1-Score: ", metrics.f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Evaluation
# https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
# Advanced Tut:
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# Saved Model to file:
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
