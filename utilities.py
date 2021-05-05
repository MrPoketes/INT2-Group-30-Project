import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Prints the statistics of the training
def diagnosis(history):
    # Loss
    plt.subplot(211)
    plt.title("Cross entropy loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="green", label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    # Accuracy
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# Download / load dataset
def load_data():
    cifar10_df = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10_df.load_data()
    # Transform labels to 1 hot encodings
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # Convert images to have a range between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = tf.keras.utils.normalize(train_images)
    test_images = tf.keras.utils.normalize(test_images)
    return train_images, train_labels, test_images, test_labels


def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        # preprocessing_function= tf.keras.applications.vgg16.preprocess_input,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.1,
        width_shift_range=0.1,
        zoom_range=0.1,
    ).flow(train_images, train_labels, batch_size=32, shuffle=True)

    return datagen


def get_predictions(model, test_images, test_labels):
    y_predict = model.predict(test_images)
    y_predict_classes = np.argmax(y_predict, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    print(tf.math.confusion_matrix(y_true, y_predict_classes))


def load_model(checkpoint_path, model):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_dir).expect_partial()

    return model


def train_model(model, train_images, train_labels, test_images, test_labels, EPOCHS):

    data = augment_data(train_images, train_labels)

    # Model saving
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, save_weights_only=True, verbose=1
    )

    return model.fit(
        x=data,
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
        steps_per_epoch=len(train_images) / 32,
        # callbacks=[cp_callback]
    )


# Runs the model
def run_model(model, EPOCHS):
    train_images, train_labels, test_images, test_labels = load_data()

    # Model summary

    model.summary()

    # Train model

    history = train_model(
        model, train_images, train_labels, test_images, test_labels, EPOCHS
    )

    # model = load_model("training_1/cp.ckpt", model)

    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)
    get_predictions(model, test_images, test_labels)

    model.save_weights("training_1/cp.ckpt")
