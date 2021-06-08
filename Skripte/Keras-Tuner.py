from imblearn.over_sampling._smote.base import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')
from get_data_for_DL import get_data, use_adasyn, use_smote, encode_labels, decode_labels
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

X_train, X_test, y_train, y_test = get_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\30_Transition_multiclass')
X_train, X_test, y_train, y_test = use_smote(X_train, X_test, y_train, y_test)
# Preparing numerical labels, Keras does not allow strings!
y_train = encode_labels(y_train)
y_test_en = encode_labels(y_test)

# Dokumentation:
# https://keras-team.github.io/keras-tuner/
# https://www.machinecurve.com/index.php/2020/06/09/automating-neural-network-configuration-with-keras-tuner/

def build_model(hp):
    dl = Sequential()
    dl.add(Dense(75, input_dim=150, activation='sigmoid')) # 'tanh', oder 'sigmoid' oder relu
    dl.add(Dense(38, activation='sigmoid'))
    dl.add(Dense(19, activation='sigmoid'))
    dl.add(Dense(10, activation='sigmoid'))
    dl.add(Dense(4,activation='softmax'))
    #dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    dl.compile(loss='categorical_crossentropy',
                optimizer=Adam(
                  hp.Choice('learning_rate',
                            values=[1e-2, 1e-3, 1e-4])),
                metrics=['accuracy'])
    return dl

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='tuning_dir',
    project_name='machinecurve_example')

tuner.search_space_summary()

# Perform random search
tuner.search(X_train, y_train,
             epochs=5,
             validation_split=0.2)