import tensorflow as tf
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers.core import Activation, Dense, Dropout, Flatten

# Create the model
def create_model():
    model = tf.keras.models.Sequential(
        [
            Conv2D(
                16, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
            ),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Activation("relu"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Activation("relu"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(1024, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            BatchNormalization(),
            Conv2D(2048, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            BatchNormalization(),
            Flatten(),
            Dropout(0.2),
            Dense(
                4096,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(
                2048,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(
                1024,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(
                512,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(
                256,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(
                128,
                activation="relu",
                bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                kernel_constraint=maxnorm(3),
            ),
            Dropout(0.2),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
