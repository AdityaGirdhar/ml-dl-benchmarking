import tensorflow as tf
import time
import numpy as np
import pandas as pd

# reading the data
train_data = pd.read_csv('../train_data.csv')
test_data = pd.read_csv('../test_data.csv')

# convert to numpy array
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# standardize the features
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Splitting data into features and target
x_data_np = train_data[:, :-1]
y_data_np = train_data[:, -1].reshape(-1, 1)

x_data_test_np = test_data[:, :-1]
y_data_test_np = test_data[:, -1].reshape(-1, 1)

# Convert numpy arrays to TensorFlow tensors
x_data = tf.constant(x_data_np, dtype=tf.float64)
y_data = tf.constant(y_data_np, dtype=tf.float64)

x_data_test = tf.constant(x_data_test_np, dtype=tf.float64)
y_data_test = tf.constant(y_data_test_np, dtype=tf.float64)

# Define the model
def model(x):
    return tf.matmul(x, w) + b

# Mean squared error loss function
def mse(t1, t2):
    diff = t1 - t2
    return tf.reduce_sum(diff * diff) / len(diff)

num_of_epochs = 1000
learning_rate = 0.001
input_size = x_data.shape[1]  # Number of features in the input data

# Initialize the weight tensor with appropriate shape
np.random.seed(42)
w_np = np.random.randn(input_size, 1)
b_np = np.random.randn(1)

# Convert numpy arrays to TensorFlow tensors
w = tf.Variable(w_np, dtype=tf.float64)
b = tf.Variable(b_np, dtype=tf.float64)
preds = model(x_data)
loss = mse(preds, y_data)
start_time = time.time()

for epoch in range(num_of_epochs):
    with tf.GradientTape() as tape:
        preds = model(x_data)
        loss = mse(preds, y_data)
    
    # Calculate gradients and update weights
    gradients = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])
    

end_time = time.time()
print('loss {}'.format(loss))
# Print the elapsed time
elapsed_time = end_time - start_time
print("Time in TensorFlow: ", elapsed_time, " seconds")

# Making predictions on training data
train_preds = model(x_data)
train_acc = mse(y_data, train_preds)
print("Training Accuracy: {}".format(train_acc.numpy()))

# Making predictions on testing data
test_preds = model(x_data_test)
test_acc = mse(y_data_test, test_preds)
print("Testing Accuracy: {} ".format(test_acc.numpy()))
