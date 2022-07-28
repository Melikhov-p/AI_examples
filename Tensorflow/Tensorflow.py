from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation, BatchNormalization, AveragePooling2D
import numpy as np
import tensorflow as tf

model = Sequential()
model.add(Dense(6, input_dim=2, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam')
# print(model.summary())
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model.fit(X, y, epochs=1000, batch_size=1)

print(model.predict(np.array([[0, 1]])), model.predict(np.array([[1, 1]])))
