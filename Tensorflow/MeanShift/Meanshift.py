import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
from memory_profiler import profile


def run_model():
  @profile
  def train_model():
      # Defining the training loop
      profiler = cProfile.Profile()
      profiler.enable()
      for epoch in range(num_of_epochs):
          tf_image2 = tf_image.numpy().copy()
          for w in range(width):
              for h in range(height):
                  k = tf.exp(tf.reduce_sum(-tf.square((tf_image[w, h] - tf_image) / bandwidth), axis=2))
                  a = tf.reduce_sum((tf_image - tf_image[w, h]) * tf.reshape(k, (width, height, 1)), axis=(0, 1))
                  b = tf.reduce_sum(k, axis=(0, 1))
                  tf_image2[w, h] = tf_image[w, h] + a / b
          tf_image.assign(tf_image2)
      profiler.disable()
      stats = pstats.Stats(profiler)
      stats.strip_dirs()
      stats.sort_stats('tottime')
      stats.print_stats('fit')

      

  @profile
  def predict():
      # Generating prediction using the trained model
      profiler = cProfile.Profile()
      profiler.enable()
      generated_image = tf_image.numpy()
      profiler.disable()
      stats = pstats.Stats(profiler)
      stats.strip_dirs()
      stats.sort_stats('tottime')
      stats.print_stats('predict')


      # Displaying the generated image
      plt.imshow(generated_image)
      plt.axis('off')
      plt.show()

  physical_devices = tf.config.list_physical_devices('GPU')
  if len(physical_devices) > 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    
  train_model()
  predict()
run_model()

