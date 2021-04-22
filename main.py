import tensorflow as tf
import numpy as np
import matplotlib as plt

# Download dataset

cifor10_df = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifor10_df.load_data()

print(train_images)
