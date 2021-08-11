from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
import pandas as pd
from copy import deepcopy
from random import shuffle
import random
from typing import Tuple

class Data:

    def __init__(self, populations: list, direc: str):
        """
        :param populations(list) - List with Population names
        :param direc - path(str) to directory with data
        """
        self.populations = populations
        self.dataframes = []
        for i in range(len(populations)):
            self.dataframes.append(pd.read_csv(direc + '\\{}.csv'.format(populations[i])))

    def random_split(self, split_ratio: float=0.2, randomState: int=None, remove_day4: bool=True):
        """
        Taking random samples for training/Test with the train_test_split function by Scikit learn
        for each population
        :param Split-ratio (float) - Ratio of Training/Test Split
        :param randomState (int) - Seed
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        """
        X = []
        y = []

        for df in self.dataframes:
            for index, row in df.iterrows():
                if remove_day4 and eval(row[1])[0] != 4:
                    X.append(row[2:-1].tolist())
                    y.append(row[-1])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=randomState)
        X, y = None, None

    def split_population_wise(self, n:int, remove_day4: bool=True):
        """
        This splitter-function takes n random populations for testing and all remaining for training.
        Those n populations are taken from the provided list at the beginning (init)
        :param n (int) - number of populations used for training
        :param remove_day4 (bool) - removes day4 trials, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = []      
        
        # Splitting
        dfs = deepcopy(self.dataframes)
        shuffle(dfs)
        df_test = dfs[:n]
        df_train = dfs[n:]

        for df in df_train:
            X, x, Y, y = self.__split_df(df, 0.0, remove_day4, False)
            X_train = X_train + X
            y_train = y_train + Y

        for df in df_test:
            X, x, Y, y = self.__split_df(df, 0.0, remove_day4, False)
            X_test = X_test + X
            y_test = y_test + Y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

    def print_shape(self):
        """
        Prints shape of training/test data
        """
        print("X_train: ", self.X_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_train: ", self.y_train.shape)
        print("y_test: ", self.y_test.shape)

    def split_trial_wise(self, split_ratio: float=0.2, remove_day4: bool=True, shuffle: bool=True):
        """
        Each Population has ~20 repetitions per trial. This function splits each of those repetitions so that 
        training and test data have some repetitions (split_ratio)
        :param Split-ratio (float) - Ratio of Training/Test Split, default is 0.2
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

        for df in self.dataframes:
            X, x, Y, y = self.__split_df(df, 1 - split_ratio, remove_day4, shuffle)
            X_train = X_train + X
            y_train = y_train + Y
            X_test = X_test + x
            y_test = y_test + y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

    def split_day_wise(self, day:int=3):
        """
        Splits each Populations into training/test data, while day x is only used for testing
        :param day (int) - default is day3 that is used for testing
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 
        for df in self.dataframes:
            header = set(df['label'].tolist())
            for trial in header:
                # geting rows with (day, Trail)-label
                rows = df.loc[df['label'] == trial].to_numpy()
                # getting response label
                response = rows[0][-1]
                # getting the actual data from the matrix
                rows = np.delete(rows, np.s_[0,1,-1], axis=1)
                for i in range(len(rows)):
                    if eval(trial)[0] == day:
                        X_test.append(rows[i])
                        y_test.append(response)
                    else:
                        X_train.append(rows[i])
                        y_train.append(response)

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)  

    def split_trial_wise_with_concat_vectors(self, n_vec: int, split_ratio: float=0.2, remove_day4: bool=True, shuffle: bool=True):
        """
        Each Population has ~20 repetitions per trial. This function splits each of those repetitions so that 
        training and test data have some repetitions (split_ratio).
        variable number of repetitions will be concatinated.
        :param n_vec (int) - number of repetiotions that are concatinated to one trial
        :param Split-ratio (float) - Ratio of Training/Test Split, default is 0.2
        :param remove_day4 (bool) - True removes day 4 trials, default is True
        :param shuffle (bool) - shuffles trials before splitting them, default is True
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

        for df in self.dataframes:
            X, x, Y, y = self.__split_df(df, 1 - split_ratio, remove_day4, shuffle, n_vec=n_vec)
            X_train = X_train + X
            y_train = y_train + Y
            X_test = X_test + x
            y_test = y_test + y

        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)       

    def __split_df(self, df:pd.DataFrame, ratio:float, rem_day4:bool, shuffle:bool, n_vec: int=1) -> Tuple[list, list, list, list]:
        """
        returns Training/Test data as lists
        """
        X_test = []
        X_train = []   
        y_test = []   
        y_train = [] 

        header = df['label'].tolist()
        responses = df['response'].tolist()
        # Removing Day 4
        trails = set()
        for i in range(len(header)):
            if rem_day4 and responses[i] == "0":
                pass
            else:
                trails.add(header[i])
            
        header = trails

        # Getting all the matrices from the trials
        for trial in header:
            # geting rows with (day, Trail)-label
            rows = df.loc[df['label'] == trial].to_numpy()
            # getting response label
            response = rows[0][-1]
            # getting the actual data from the matrix
            rows = np.delete(rows, np.s_[0,1,-1], axis=1)
            if shuffle:
                # shuffle PC-Matrix
                np.random.shuffle(rows)

            if n_vec == 1:
                pass
            else:
                new_rows = []
                # taking samples
                while len(rows) > n_vec:
                    vecs = rows[:n_vec]
                    # deleting vectors that are already taken
                    rows = rows[n_vec:]
                    # Concat vectors to one
                    new_rows.append(np.concatenate(vecs))
                rows = new_rows

            # Splitting into Test and training
            cut = int(ratio*len(rows))
            for i in range(len(rows)):
                if i < cut or ratio == 0.0:
                    X_train.append(rows[i])
                    y_train.append(response)
                else:
                    X_test.append(rows[i])
                    y_test.append(response)

        return X_train, X_test, y_train, y_test

    def shuffle_labels(self):
        """
        shuffles labels to have a (random) Benchamrk
        """
        random.shuffle(self.y_train)
        random.shuffle(self.y_test)

    def use_SMOTE(self):
        """performs SMOTE on training data"""
        smote = SMOTE()
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def use_ADASYN(self):
        """performs ADASYN on training data"""
        ada = ADASYN()
        self.X_train, self.y_train = ada.fit_resample(self.X_train, self.y_train)


    def get_data(self):
        """
        returns X_train, X_test, etc
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def k_fold_cross_validation_populationwise(self, K: int=5, rem_day4:bool=True, smote:bool=True, shuffle: bool=False) -> dict:
        """
        performs k-fold cross validation, 
        returns dict: {Population: {K-Fold: {X:.., y:..}}}
        """
        counter = 0
        results = {}
        for df in self.dataframes:
            k_folds = {}

            header = set(df['label'].tolist())
            # Removing Day 4
            trails = set()
            for i in header:
                trail = eval(i)
                if trail[0] != 4:
                    trails.add(i)
                else:
                    if not(rem_day4):
                        trails.add(i)

            header = trails

            # Getting all the matrices from the trials
            for k in range(K):
                k_folds[k] = {"X_train": [], "X_test": [], "y_test": [], "y_train": []}
                for trial in header:
                    # geting rows with (day, Trail)-label
                    rows = df.loc[df['label'] == trial].to_numpy()
                    # getting response label
                    response = rows[0][-1]
                    # getting the actual data from the matrix
                    rows = np.delete(rows, np.s_[0,1,-1], axis=1)

                    chunks = np.array_split(rows, K)
                    for chunk in chunks[k]:
                        k_folds[k]["X_test"].append(chunk.astype(np.float))
                        k_folds[k]["y_test"].append(response)

                    train_chunks = np.delete(chunks, k, axis=0)
                    for chunk in train_chunks:
                        for ch in chunk:
                            k_folds[k]["X_train"].append(ch.astype(np.float))
                            k_folds[k]["y_train"].append(response)

            for k in range(K):

                self.X_test = np.asarray(k_folds[k]["X_test"])
                self.y_test = np.asarray(k_folds[k]["y_test"])
                self.X_train = np.asarray(k_folds[k]["X_train"])
                self.y_train = np.asarray(k_folds[k]["y_train"])

                if smote:
                    self.use_SMOTE()

                if shuffle:
                    self.shuffle_labels()

                k_folds[k]["X_test"] = self.X_test
                k_folds[k]["X_train"] = self.X_train
                k_folds[k]["y_test"] = self.y_test
                k_folds[k]["y_train"] = self.y_train
            
            results[self.populations[counter]] = k_folds
            counter +=1

        return results
        
    def k_fold_cross_validation(self, K: int=5, rem_day4:bool=True, smote: bool=True, shuffle: bool=False) -> dict:
        """
        performs k-fold cross validation on dataset
        uses all Populations at once
        """
        k_folds = {}

        for k in range(K):
            k_folds[k] = {"X_train": [], "X_test": [], "y_test": [], "y_train": []}
            for df in self.dataframes:
                header = set(df['label'].tolist())
                # Removing Day 4
                trails = set()
                for i in header:
                    trail = eval(i)
                    if trail[0] != 4:
                        trails.add(i)
                    else:
                        if not(rem_day4):
                            trails.add(i)

                header = trails

                for trial in header:
                    # geting rows with (day, Trail)-label
                    rows = df.loc[df['label'] == trial].to_numpy()
                    # getting response label
                    response = rows[0][-1]
                    # getting the actual data from the matrix
                    rows = np.delete(rows, np.s_[0,1,-1], axis=1)

                    chunks = np.array_split(rows, K)
                    for chunk in chunks[k]:
                        k_folds[k]["X_test"].append(chunk.astype(np.float))
                        k_folds[k]["y_test"].append(response)

                    train_chunks = np.delete(chunks, k, axis=0)
                    for chunk in train_chunks:
                        for ch in chunk:
                            k_folds[k]["X_train"].append(ch.astype(np.float))
                            k_folds[k]["y_train"].append(response)


        for k in range(K):

            self.X_test = np.asarray(k_folds[k]["X_test"])
            self.y_test = np.asarray(k_folds[k]["y_test"])
            self.X_train = np.asarray(k_folds[k]["X_train"])
            self.y_train = np.asarray(k_folds[k]["y_train"])

            if smote:
                    self.use_SMOTE()

            if shuffle:
                    self.shuffle_labels()

            k_folds[k]["X_test"] = self.X_test
            k_folds[k]["X_train"] = self.X_train
            k_folds[k]["y_test"] = self.y_test
            k_folds[k]["y_train"] = self.y_train

        return k_folds
