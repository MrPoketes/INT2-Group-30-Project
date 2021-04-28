import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os

# Global variables
EPOCHS = 1

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

def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
    preprocessing_function= tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    horizontal_flip=True
    ).flow(train_images, train_labels, batch_size=32, shuffle=True)

    return datagen


# Create the model
def create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu", bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)),
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

def load_model(checkpoint_path, model):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_dir).expect_partial()

    return model

def train_model(model, train_images, train_labels, test_images, test_labels):   

    data = augment_data(train_images, train_labels)    

    # Model saving 
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,save_weights_only=True, verbose=1)

    return model.fit(
        x=data,
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
        steps_per_epoch=len(train_images) / 32,
        callbacks=[cp_callback]
    )
    
# Runs the model
def run_model():
    train_images, train_labels, test_images, test_labels = load_data()
    model = create_model()
    # Model summary

    model.summary()

    # Fit model
    history = train_model(model, train_images, train_labels, test_images, test_labels)

    #model = load_model("training_1/cp.ckpt", model)
  
    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)

    #model.save_weights("training_1/cp.ckpt")


run_model()