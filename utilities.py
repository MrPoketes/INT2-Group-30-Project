import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Prints the statistics of the training
def diagnosis(history):
    # Loss
    figures, axis = plt.subplots(2,1)
    axis[0].plot(history.history["loss"], color="blue", label="train")
    axis[0].plot(history.history["val_loss"], color="green", label="test")
    axis[0].set_title("Cross entropy loss")
    # Accuracy
    axis[1].set_title("Accuracy")
    axis[1].plot(history.history["accuracy"], label="train")
    axis[1].plot(history.history["val_accuracy"], label="test")
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


def get_predictions(model, test_images, test_labels):
    y_predict = model.predict(test_images)
    y_predict_classes = np.argmax(y_predict, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    print(tf.math.confusion_matrix(y_true, y_predict_classes))


# Runs the model
def run_model(model, EPOCHS):
    train_images, train_labels, test_images, test_labels = load_data()
    # Model summary
    model.summary()
    # Train model
    history = model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
    )
    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)
    get_predictions(model, test_images, test_labels)
