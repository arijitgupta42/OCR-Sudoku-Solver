import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

#load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1).astype('float32')/255

series = pd.Series(train_labels)
train_labels = pd.get_dummies(series).to_numpy()

#build the model
model = tf.keras.models.Sequential()
#add convolutional layers with input that matches our dataset
model.add(tf.keras.layers.Conv2D(254, kernel_size=(3,3), input_shape=(28,28, 1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
#convert from 2D input to 1D vectors
model.add(tf.keras.layers.Flatten())
#finish our model with densely connected layers
model.add(tf.keras.layers.Dense(140, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(80, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
#output layer with 10 units (one per each class 0-9)
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_images,
          train_labels, 
          epochs=5,
          validation_data = [test_images.reshape(10000,28,28,1).astype('float32')/255,
                             pd.get_dummies(pd.Series(test_labels)).to_numpy()])

#save the model to use it to make predictions
model.save('./models/my_model.h5')