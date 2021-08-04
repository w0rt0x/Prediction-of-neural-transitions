# tutorials used:
# https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from get_data_for_DL import get_matrix_data, encode_labels, decode_labels, use_smote
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
import random
"""
X_train, X_test, y_train, y_test = get_matrix_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\100_Transition_multiclass')
X_train = np.concatenate((X_train, X_test))
y_train = encode_labels(np.concatenate((y_train, y_test)))
X1, X2, y1, y2 = get_matrix_data(['bl684_no_white_Pop03','bl689-1_one_white_Pop09','bl688-1_one_white_Pop05', 'bl709_one_white_Pop11', 'bl660-1_two_white_Pop07'], path=r'D:\Dataframes\100_Transition_multiclass')
X_test = np.concatenate((X1, X2))
y_test = np.concatenate((y1, y2))
y_test_en = encode_labels(np.concatenate((y1, y2)))

X_train, X_test, y_train, y_test_en = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test_en)
X_train = np.reshape(X_train, X_train.shape + tuple([1]))
X_test = np.reshape(X_test, X_test.shape + tuple([1]))
"""

X_train, X_test, y_train, y_test = get_matrix_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], path=r'D:\Dataframes\most_active_neurons\40')
#random.shuffle(y_train)
#random.shuffle(y_test)
y_train = encode_labels(y_train)
y_test_en = encode_labels(y_test)

X_train, X_test, y_train, y_test_en = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test_en)
X_train = np.reshape(X_train, X_train.shape + tuple([1]))
X_test = np.reshape(X_test, X_test.shape + tuple([1]))

print(X_train.shape)
print(X_test.shape)
# https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a

model = Sequential() 
# 32 = Output Filter, (3,3) = Kernel Size, (20,30,1) Höhe Breite Tiefe
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (10,40,1))) 
#32 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
# choose the best features via MAX pooling
model.add(MaxPooling2D(pool_size = (2, 2))) 
# randomly turn neurons on and off to improve convergence: Dropout Layer
model.add(Dropout(0.25)) 
# flatten since too many dimensions, we only want a classification output 
model.add(Flatten()) 
# fully connected to get all relevant data
model.add(Dense(128, activation = 'relu')) 
# Another Dropout Layer
model.add(Dropout(0.5)) 
# output a softmax to squash the matrix into output probabilities
model.add(Dense(4, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 5, epochs = 100, verbose = 1, validation_data = (X_test, y_test_en))
score = model.evaluate(X_test, y_test_en, verbose = 2) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

y_pred = decode_labels(model.predict_classes(X_test))
print("F1-Score wighted: ", metrics.f1_score(y_test, y_pred,average='weighted'))
print("F1-Score macro: ", metrics.f1_score(y_test, y_pred,average='macro'))
print("F1-Score micro: ", metrics.f1_score(y_test, y_pred,average='micro'))
print(classification_report(y_test, y_pred))
