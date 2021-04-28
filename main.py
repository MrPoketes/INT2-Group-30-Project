import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Global variables
EPOCHS = 50

# Download / load dataset
def load_data():
    cifor10_df = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifor10_df.load_data()
    # Transform labels to 1 hot encodings
    train_labels = tf.one_hot(train_labels, 1)
    test_labels = tf.one_hot(test_labels, 1)

    # Convert images to have a range between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels


# Create the model
def create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(32, 32, 3)
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
            ),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# Prints the statistics of the training
def diagnosis(history):
    # Loss
    plt.subplot(211)
    plt.title("Cross entropy loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="green", label="test")
    plt.legend()
    plt.show()
    # Accuracy
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.legend()
    plt.show()


# Runs the model
def run_model():
    train_images, train_labels, test_images, test_labels = load_data()
    model = create_model()
    # Model summary
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
    )
    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)


run_model()
