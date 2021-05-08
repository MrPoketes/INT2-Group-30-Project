import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from models.lewis_model import create_model

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

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 0.0000001)
    test_images = (test_images - mean) / (std + 0.0000001)
    return train_images, train_labels, test_images, test_labels


def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.1,
        width_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest"
    ).flow(train_images, train_labels, batch_size=32, shuffle=True)

    return datagen


def get_predictions(model, test_images, test_labels):
    y_predict = model.predict(test_images)
    y_predict_classes = np.argmax(y_predict, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    print(tf.math.confusion_matrix(y_true, y_predict_classes))

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    print(len(y_predict_classes))

    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[y_predict_classes[i]] + "/" + class_names[y_true[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.show()

    return y_predict_classes


def load_model(checkpoint_path):
    train_images, train_labels, test_images, test_labels = load_data()
    
    model = create_model(train_images.shape[1:])
    model.load_weights(checkpoint_path).expect_partial()

    evaluate(model, test_images, test_labels)

    get_predictions(model, test_images, test_labels)

    return model

def evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")

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
        steps_per_epoch=len(train_images) / 32
    )


# Runs the model
def run_model(EPOCHS):
    train_images, train_labels, test_images, test_labels = load_data()

    model = create_model(train_images.shape[1:])

    # Model summary
    model.summary()

    # Train model
    history = train_model(
        model, 
        train_images, 
        train_labels, 
        test_images, 
        test_labels, EPOCHS
    )

    # model = load_model("training_1/cp.ckpt", model)

    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)
    get_predictions(model, test_images, test_labels)

    model.save_weights("training_1/cp.ckpt")

