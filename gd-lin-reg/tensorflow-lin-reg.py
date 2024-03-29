import numpy as np
import pandas as pd
import time
import tensorflow as tf
import psutil

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y, learning_rate=0.01, num_iterations=1000):
        #start_time = time.time()
        
        # Normalize the features using z-score normalization
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Add a column of ones to X for the bias term
        X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
        
        # Convert the numpy arrays to TensorFlow tensors
        X_tensor = tf.constant(X_normalized, dtype=tf.float32)
        y_tensor = tf.constant(y.reshape(-1, 1), dtype=tf.float32)
        
        num_samples, num_features = X_tensor.shape
        
        # Initialize weights with zeros
        self.weights = tf.Variable(tf.zeros((num_features, 1), dtype=tf.float32))

        start_time = time.time()
        # Perform gradient descent
        for i in range(num_iterations):
            predictions = tf.matmul(X_tensor, self.weights)
            error = predictions - y_tensor
            gradients = tf.matmul(tf.transpose(X_tensor), error) / num_samples
            self.weights.assign_sub(learning_rate * gradients)
        
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time
    
    def predict(self, X):
        start_time = time.time()
        
        # Normalize the features using z-score normalization
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Add a column of ones to X for the bias term
        X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
        
        # Convert the numpy array to a TensorFlow tensor
        X_tensor = tf.constant(X_normalized, dtype=tf.float32)
        
        # Make predictions
        predictions = tf.matmul(X_tensor, self.weights)
        
        end_time = time.time()
        execution_time = end_time - start_time
        return predictions, execution_time

# Specify the device type as "cuda"
device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")
print("Device:", "cuda" if tf.config.list_physical_devices('GPU') else "cpu")

# Load and preprocess the data
data = np.loadtxt(r'custom_2017_2020.csv', delimiter=',', skiprows=1)
cols = ["exp_imp", "Year", "month", "ym", "Country", "Custom", "hs2", "hs4", "hs6", "hs9", "Q1", "Q2", "Value"]
df = pd.DataFrame(data, columns=cols)
train = df.sample(frac=0.8)
test = df.sample(frac=0.2)

# Split the data into features and target
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# Create an instance of LinearRegression class
lr = LinearRegression()

# Measure memory usage during fit
fit_memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
print("Memory usage during fit:", fit_memory_usage, "MB")

# Fit the model
fit_execution_time = lr.fit(X, y)
print("Time to fit the model:", fit_execution_time)

# Measure memory usage during predict
predict_memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
print("Memory usage during predict:", predict_memory_usage, "MB")

# Make predictions
predictions = lr.predict(X_test)

# Calculate MSE loss
mse_loss = np.mean((predictions[0].numpy().flatten() - y_test) ** 2)
print("MSE Loss:", mse_loss)
print("Time to predict:", predictions[1])