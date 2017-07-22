"""
__name__ = digits_keras
__author__ = Yash Patel
__description__ = MNIST implementation with Keras
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

def create_model():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
	x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

	y_train = np_utils.to_categorical(y_train)
	y_test  = np_utils.to_categorical(y_test)

	model = Sequential()
	model.add(Convolution2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
	model.add(Convolution2D(64, (5,5)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Dropout(.50))
	model.add(Dense(10))

	model.compile(loss='categorical_crossentropy',
		optimizer='Adadelta',
		metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=128, epochs=10)
	score = model.evaluate(x_test, y_test, verbose=0)
	print(score)

if __name__ == "__main__":
	create_model()