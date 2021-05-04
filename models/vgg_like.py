import tensorflow as tf
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import (
    RandomFlip,
    RandomCrop,
    RandomRotation,
    RandomZoom,
)

# Create the model
def create_model():
    model = tf.keras.models.Sequential(
        [
            RandomCrop(32, 32, input_shape=(32, 32, 3)),
            RandomFlip(mode="horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)
            ),
            BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.AveragePooling2D((1, 1), strides=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model