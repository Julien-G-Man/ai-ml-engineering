import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "tf_model_v1.h5"

# Sample datasets
(X_train, y_train), _ = mnist.load_data()

X_train = X_train.reshape(-1, 28*28) / 255.0

model = Sequential([
    Dense(128, activation='relu', input_shape=(784, )),
    Dense(10,  activation='softmax')
])


def main():
    model.compile(
        optimizer='adam', 
        metrics=['accuracy'],
        loss='sparse_categorical_crossentropy'
    )
    model.fit(X_train, y_train, epochs=5)
    model.save(MODEL_PATH)
    
    
    
def run_loaded_model(path):
    loaded_model = load_model(MODEL_PATH)
    loaded_model.predict(X_train)
