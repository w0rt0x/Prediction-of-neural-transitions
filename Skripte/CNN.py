# tutorials used:
# https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from get_data_for_DL import get_matrix_data

X_train, X_test, y_train, y_test = get_matrix_data(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'])

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

print(X_train[0])
