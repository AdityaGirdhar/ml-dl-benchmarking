import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from memory_profiler import profile
# Load the dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Normalize and preprocess the data
train_data = train_data.reshape((60000, 28 * 28))
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28 * 28))
test_data = test_data.astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the neural net using Keras Sequential API
model = models.Sequential()
model.add(layers.Dense(200, activation='relu', input_shape=(784,)))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with the SGD optimizer
learning_rate = 0.1
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary (optional)
model.summary()

# Set up the batch size and number of epochs
batch_size = 100
num_of_epochs = 10

# Training function
@profile   # Add memory_profiler decorator
def training_function():
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=1, verbose=0)

# Run the training function once before starting the profiler
training_function()

# Start memory profiling


# Start time profiling using cProfile
import cProfile
pr = cProfile.Profile()
pr.enable()

# Train the model for multiple epochs
for epoch in range(num_of_epochs):
    training_function()

# Stop profiling
pr.disable()

# Print the profiling results
print("Time profiling results:")
pr.print_stats(sort="time")

loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print("Model accuracy on test data:", accuracy)

# Measure prediction time
import cProfile
pr = cProfile.Profile()
pr.enable()

predictions = model.predict(test_data)
pr.disable()

# Print the profiling results
print("Prediction Time profiling results:")
pr.print_stats(sort="time")

