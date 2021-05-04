import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.advanced_activations import ReLU
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
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=(32, 32, 3)),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.MaxPooling2D((3, 2), strides=1),
            Dropout(0.2),
            tf.keras.layers.Conv2D(128, (5, 5), padding="same"),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.MaxPooling2D((3, 2)),
            Dropout(0.4),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
            BatchNormalization(),
            ReLU(),
            tf.keras.layers.MaxPooling2D((3, 2)),
            Dropout(0.5),
            # tf.keras.layers.AveragePooling2D((1, 1), strides=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.006, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model