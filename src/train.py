import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow_datasets as tfds


def download_dataset(dset):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dset, 'data', unzip=True)

def create_model(shape=(256, 256, 3)):
    inputs = keras.Input(shape)
    conv1 = layers.Conv2D(64, 3)(inputs)
    conv1 = layers.MaxPooling2D()(conv1)
    conv1 = layers.Flatten()(conv1)
    dense1 = layers.Dense(128)(conv1)
    dense2 = layers.Dense(16)(dense1)
    out = layers.Dense(1)(dense2)
    model = keras.Model(inputs, out)
    model.summary()
    return model

def train_model(model):
    data = tf.keras.utils.image_dataset_from_directory("data")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
    model.fit(x=data, epochs=10) # why the hell do I have to do this when it advertises dictionary support

    return model




def main():
    model = create_model()
    model = train_model(model)


if __name__ == "__main__":
    main()