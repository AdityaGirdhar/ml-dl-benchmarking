import tensorflow as tf
import numpy as np
from sklearn.metrics import silhouette_score
import time

def mean_shift_clustering(data, bandwidth, num_iterations, width, height):
    print(" started")
    start = time.time()
    for _ in range(num_iterations):
        updated_data = tf.Variable(data.numpy(), dtype=tf.float32)  # Create a copy of data to update
        for w in range(width):
            for h in range(height):
                k = tf.exp(tf.reduce_sum(-tf.square((data[w, h] - data) / bandwidth), axis=2))
                a = tf.reduce_sum((data - data[w, h]) * tf.reshape(k, (width, height, 1)), axis=(0, 1))
                b = tf.reduce_sum(k, axis=(0, 1))
                updated_data[w, h].assign(data[w, h] + a / b)  # Use assign to update the values
        data.assign(updated_data)  # Assign the updated values back to the original data
    endtime = time.time()
    print("ended ", endtime - start, " ")
    return data

def calculate_silhouette_score(data, labels):
    return silhouette_score(data.numpy().reshape(-1, data.shape[-1]), labels)

def run_model():
    img = tf.keras.preprocessing.image.load_img('peppers.png', target_size=(150, 200))
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

    # Perform Mean Shift clustering
    clustered_image = mean_shift_clustering(tf_image, bandwidth, num_of_epochs, width, height)

    # Calculate silhouette score
    flat_image = tf.reshape(clustered_image, [-1, colors])
    labels = tf.argmin(tf.norm(flat_image[:, None] - flat_image, axis=-1), axis=-1)
    silhouette_score_val = calculate_silhouette_score(flat_image, labels)
    print("Silhouette Score:", silhouette_score_val)

run_model()
