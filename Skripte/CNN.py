# tutorials used:
# https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from get_data_for_DL import get_matrix_data, encode_labels, decode_labels
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics

X_train, X_test, y_train, y_test = get_matrix_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'])
y_train = encode_labels(y_train)
y_test_en = encode_labels(y_test)

print(type(X_train))
print(type(X_train[0]))
print(type(X_train[0][0]))
print(type(y_train))
print(type(y_train[0]))

model = Sequential() 
# 32 = Output Filter, (3,3) = Kernel Size, (20,30,1) Höhe Breite Tiefe
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (20, 30, 1))) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 5, epochs = 100, verbose = 1, validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test, verbose = 2) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

y_pred = decode_labels(model.predict_classes(X_test))
print("F1-Score wighted: ", metrics.f1_score(y_test, y_pred,average='weighted'))
print("F1-Score macro: ", metrics.f1_score(y_test, y_pred,average='macro'))
print("F1-Score micro: ", metrics.f1_score(y_test, y_pred,average='micro'))
print(classification_report(y_test, y_pred))
