from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
import random
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

def get_data(pops, path=r'r"D:\Dataframes\20PCs', ratio=0.8, n=10, remove_day4=True):
    X_test = []
    X_train = []   
    y_test = []   
    y_train = []      

    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
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
            response = rows[0][-1]  # 1 if (rows[0][-1] > 0) else 0
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
                    X_train.append(data[i].tolist())
                    y_train.append(response)
                else:
                    X_test.append(data[i].tolist())
                    y_test.append(response)

    return X_train, X_test, y_train, y_test

def getMeanAndVariance(pops, path=r'r"D:\Dataframes\20PCs', ratio=0.8, n=10, remove_day4=True):
    """
    returns mean and variance of matrices
    """
   
    data = []
    responses = []
    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
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
            response = rows[0][-1]  # 1 if (rows[0][-1] > 0) else 0
            # getting PC-Matrix and shuffeling PC-Arrays randomly
            rows = np.delete(rows, np.s_[0,1,-1], axis=1)
            for j in range(len(rows)):
                mean = np.mean(rows[j])
                var = np.var(rows[j])
                data.append([mean, var])
                responses.append(response)

    return np.asarray(data), np.asarray(responses)

def get_matrix_data(pops, path=r'D:\Dataframes\30_Transition_multiclass', balanced=True, ratio=0.8, n=10, remove_day4=True):
    """
    Gets trails matrices, upsampling by shuffleing the rows with balanced-parameter
    """
    X_test = []
    X_train = []   
    y_test = []   
    y_train = []    

    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
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
            # geting rows with (day, Trail)-label => Matrix 20 x 30
            rows = df.loc[df['label'] == trial].to_numpy()
            matrix=[]
            for i in rows:
                matrix.append(i[2:-1])
            #print(matrix)
            # matrix needs 20 rows, up or downsampling if this is not the case
            if len(matrix) < 20:
                while len(matrix)<20:
                    matrix.append(random.choice(matrix))
            if len(matrix) > 20:
                matrix = matrix[0:20]
            for i in range(len(matrix)):
                matrix[i] = matrix[i].tolist()
            # splitting data into training and test
            """
            X_train.append(matrix[:10])
            y_train.append(rows[0][-1])
            X_test.append(matrix[10:])
            y_test.append(rows[0][-1])
            """
            #for i in range(n):
                #m = deepcopy(matrix)
                #random.shuffle(m)
            #if i < int(n*ratio):
            X_train.append(matrix[:10])
            y_train.append(rows[0][-1])
            #else:
            X_test.append(matrix[10:])
            y_test.append(rows[0][-1])
            
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
    #X_test, y_test = smote.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test

def use_adasyn(X_train, X_test, y_train, y_test):
    ada = ADASYN()
    X_train, y_train = ada.fit_resample(X_train, y_train)
    #X_test, y_test = ada.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test

def encode_labels(y):
    """ Encodes string labels"""
    y_hot = []
    table = {"0->0":[1,0,0,0], "0->1":[0,1,0,0], "1->0":[0,0,1,0], "1->1":[0,0,0,1]}
    for i in range(len(y)):
        y_hot.append(table[y[i]])
    return y_hot

def decode_labels(y):
    """ Encodes string labels"""
    y_hot = []
    table = {0:"0->0", 1:"0->1", 2:"1->0", 3:"1->1"}
    for i in range(len(y)):
        y_hot.append(table[y[i]])
    return y_hot

def random_split(pops, path=r'r"D:\Dataframes\20PCs', split_ratio=0.2, randomState=None, strat=None):
    dataframes = []
    for pop in pops:
        df = pd.read_csv(path + '\\' + pop + '.csv')
        dataframes.append(df)

    X = []
    y = []

    for df in dataframes:
        for index, row in df.iterrows():
            if eval(row[1])[0] != 4:
                X.append(row[2:-1].tolist())
                y.append(row[-1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=randomState, stratify=strat)
    X, y = None, None

    return X_train, X_test, y_train, y_test