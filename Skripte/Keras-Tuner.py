from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from get_data_for_DL import get_data, use_adasyn, use_smote, encode_labels, decode_labels
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from tensorflow.keras.layers import Flatten
import numpy as np

X_train, X_test, y_train, y_test = get_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\100_Transition_multiclass')
X_train, X_test, y_train, y_test = use_smote(X_train, X_test, y_train, y_test)
# Preparing numerical labels, Keras does not allow strings!
X = np.asarray(X_train + X_test)
y = np.asarray(encode_labels(y_train + y_test))

# Dokumentation:
# https://keras-team.github.io/keras-tuner/
# https://www.machinecurve.com/index.php/2020/06/09/automating-neural-network-configuration-with-keras-tuner/
# https://keras-team.github.io/keras-tuner/#

class MyHyperModel(HyperModel):

    def __init__(self, input_size, num_classes):
        self.num_classes = num_classes
        self.input_size = input_size

    def build(self, hp):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_size))
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                                        activation='sigmoid'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', 
               optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
               metrics=['accuracy'])
        return model

hypermodel = MyHyperModel(input_size=(500, 1), num_classes=4)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='tuning_dir',
    project_name='machinecurve_example')

tuner.search_space_summary()

# Perform random search
tuner.search(X, y,
             epochs=100,
             validation_split=0.2)

tuner.results_summary()
