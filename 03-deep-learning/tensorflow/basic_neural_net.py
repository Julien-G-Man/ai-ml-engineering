import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Sample datasets
(X_train, y_train), _ = mnist.load_data()

X_train = X_train.reshape(-1, 28*28) / 255.0

model = Sequential([
    Dense(128, activation='relu', input_shape=(784, )),
    Dense(10,  activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)