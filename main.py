import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Global variables
EPOCHS =10 

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
    figures, axis = plt.subplots(2,1)
    axis[0].set_title("Cross entropy loss")
    axis[0].plot(history.history["loss"], color="blue", label="train")
    axis[0].plot(history.history["val_loss"], color="green", label="test")
    axis[0].legend()
    # Accuracy
    axis[1].set_title("Accuracy")
    axis[1].plot(history.history["accuracy"], label="train")
    axis[1].plot(history.history["val_accuracy"], label="test")
    axis[1].legend()
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
