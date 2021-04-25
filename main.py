import tensorflow as tf
import numpy as np
import matplotlib as plt
from keras.utils.np_utils import to_categorical

# Download / load dataset

cifor10_df = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifor10_df.load_data()

# Transform labels to 1 hot encodings
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Convert images to have a range between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
