#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 06:21:36 2020

@author: vivek
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flattening images in 3-D array form to vectorised form of 784
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels))
X_test = X_test.reshape((X_test.shape[0], num_pixels))

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encoded outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# 1 layer NN

def NN():
	
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

model=NN()

# Fitting the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Evaluation of model
score, acc = model.evaluate(X_test, y_test,
                            batch_size=200)
print('Test score:', score)
print('Test accuracy:', acc)

