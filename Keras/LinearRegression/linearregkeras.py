import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf
import cProfile
import pstats
from memory_profiler import profile

class LinearRegression:
    def __init__(self):
        self.model = None
    
    @profile
    def fit(self, X, y, epochs=1000, batch_size=32):
        # Normalize the features and target using training data statistics
        X_mean, X_std = X.mean(), X.std()
        y_mean, y_std = y.mean(), y.std()
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        
        # Define the model
        self.model = Sequential()
        self.model.add(Dense(1, input_shape=(X.shape[1],)))
        optimizer = SGD(0.01)
        
        # Compile the model
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Profile the training time
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Fit the model
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Stop profiling and print the training time
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('fit')
        train_time = stats.total_tt
        
        return train_time
    @profile
    def predict(self, X):
        # Normalize the features using training data statistics
        X_mean, X_std = X.mean(), X.std()
        X = (X - X_mean) / X_std
        
        # Profile the prediction time
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Stop profiling and print the prediction time
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('predict')
        predict_time = stats.total_tt
        
        return predictions, predict_time

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Enable memory growth for the first GPU device
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Create a sample DataFrame
df = pd.read_csv("custom_2017_2020.csv", delimiter=',', skiprows=1)
train = df.sample(frac=0.8)
test = df.sample(frac=0.2)

# Split the data into features and target
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# Create an instance of the LinearRegression class
lr = LinearRegression()

# Fit the model and measure the training time
train_time = lr.fit(X, y)
print("Training Time:", train_time)

# Make predictions and measure the prediction time
predictions, predict_time = lr.predict(X_test)
print("Prediction Time:", predict_time)

# Calculate MSE loss
mse_loss = np.mean((predictions[0] - y_test) ** 2)

# Calculate training accuracy
train_accuracy = 100 - (np.mean((predictions[0] - y) ** 2) / np.mean(y**2) * 100)

# Calculate testing accuracy
test_accuracy = 100 - (mse_loss / np.mean(y_test**2) * 100)

# Print accuracies
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", mse_loss)
