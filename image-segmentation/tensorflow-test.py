import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Loading image and pre-processing
img = tf.keras.preprocessing.image.load_img('peppers.png', target_size=(200, 150))
image = tf.keras.preprocessing.image.img_to_array(img)
image = image.astype('float32') / 255.0

# Getting the dimensions of the image and setting hyperparameters
width = image.shape[0]
height = image.shape[1]
colors = image.shape[2]
num_of_epochs = 10
bandwidth = 0.1

# Creating TensorFlow variables
tf_image = tf.Variable(image, dtype=tf.float32)

# Defining the training loop
start = time.time()
for epoch in range(num_of_epochs):
    tf_image2 = tf_image.numpy().copy()
    for w in range(width):
        for h in range(height):
            k = tf.exp(tf.reduce_sum(-tf.square((tf_image[w, h] - tf_image) / bandwidth), axis=2))
            a = tf.reduce_sum((tf_image - tf_image[w, h]) * tf.reshape(k, (width, height, 1)), axis=(0, 1))
            b = tf.reduce_sum(k, axis=(0, 1))
            tf_image2[w, h] = tf_image[w, h] + a / b
    tf_image.assign(tf_image2)
end = time.time()

# Training the model
with open('time.txt', 'a') as f:
    # Redirect stdout to the file
    sys.stdout = f
    print(f"Tensorflow: {end-start} seconds")