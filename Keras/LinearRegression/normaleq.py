# it doesnt do much everything is done using numpy so doesnt really matter
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras import backend as K
import cProfile
import pstats

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # Profile the training time
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Convert X and y to Keras tensors
        X_tensor = K.variable(X)
        y_tensor = K.variable(y)

        # Reshape y_tensor to match the shape of X_tensor
        y_tensor = K.reshape(y_tensor, (-1, 1))

        # Compute the weights using the normal equation
        X_transpose = K.transpose(X_tensor)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose), y)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('fit')
        train_time = stats.total_tt
        print("Train time is", train_time)
    
    def predict(self, X):
        # Profile the prediction time
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Make predictions
        predictions = np.dot(X, self.weights)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('predict')
        test_time = stats.total_tt
        print("Test time is", test_time)
        
        return predictions, test_time

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create a sample DataFrame
df = pd.read_csv("custom_2017_2020.csv", delimiter=',', skiprows=1)

cols = ["exp_imp", "Year", "month", "ym", "Country", "Custom", "hs2", "hs4", "hs6", "hs9", "Q1", "Q2", "Value"]
df = pd.DataFrame(df, columns=cols)
train = df.sample(frac=0.001)
test = df.sample(frac=0.0005)

# Split the data into features and target
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# Create an instance of LinearRegression class
lr = LinearRegression()

# Fit the model
lr.fit(X, y)

# Make predictions
predictions = lr.predict(X_test)

# Calculate MSE loss
mse_loss = np.mean((predictions[0] - y_test) ** 2)

# Print results
print("Time to predict:", predictions[1])
print("MSE Loss:", mse_loss)
