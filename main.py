import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import os

# Download / load dataset

cifor10_df = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifor10_df.load_data()

classes = max(train_labels) - min(train_labels) + 1

# Transform labels to 1 hot encodings
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Convert images to have a range between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.Sequential()

# Layer setup 
#------------------------------------------------------
model.add(Conv2D(32, (3, 3), input_shape=train_images.shape[1:], padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001), kernel_constraint=maxnorm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001), kernel_constraint=maxnorm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(classes))
model.add(Activation('softmax'))
#------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, verbose=1)
                                                
model.load_weights(checkpoint_path)

#model.fit(train_images, train_labels,validation_data=(test_images, test_labels), epochs=10, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)