import numpy as np
import pandas as pd
import time
import tensorflow as tf

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        start_time = time.time()
        
        # Normalize the features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_tensor = tf.constant(X, dtype=tf.float32)
        y_tensor = tf.constant(y, dtype=tf.float32)
        
        # Compute the weights using the normal equation
        X_transpose = tf.transpose(X_tensor)
        self.weights = tf.linalg.inv(X_transpose @ X_tensor) @ X_transpose @ tf.expand_dims(y_tensor, axis=1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time
    
    def predict(self, X):
        start_time = time.time()
        
        # Normalize the features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_tensor = tf.constant(X, dtype=tf.float32)
        
        # Make predictions
        predictions = X_tensor @ self.weights
        
        end_time = time.time()
        execution_time = end_time - start_time
        return predictions, execution_time

# Specify the device type as "cuda"
device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")

# Load and preprocess the data
data = np.loadtxt(r'Tensorflow\custom_2017_2020.csv',delimiter=',')
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

# Fit the model
fit_execution_time = lr.fit(X, y)
print("Time to fit the model:", fit_execution_time)

# Make predictions
predictions = lr.predict(X_test)

# Calculate MSE loss
mse_loss = np.mean((predictions[0].numpy().flatten() - y_test) ** 2)
print("MSE Loss:", mse_loss)
print("Time to predict:", predictions[1])
