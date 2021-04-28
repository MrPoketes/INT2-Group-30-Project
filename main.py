import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

# Download / load dataset

cifor10_df = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifor10_df.load_data()

# Transform labels to 1 hot encodings
"""
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
"""


# Convert images to have a range between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Model
model = models.Sequential()

# Augmentation to increase val_acurracy
model.add(layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(32, 32, 3)))
model.add(layers.experimental.preprocessing.RandomRotation(0.1))
model.add(layers.experimental.preprocessing.RandomZoom(0.1))

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=25, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_loss, test_acc)