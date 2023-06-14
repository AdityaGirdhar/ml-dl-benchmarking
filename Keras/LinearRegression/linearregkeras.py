import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import time
class LinearRegression:
    def __init__(self):
        self.model = None
    
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
        
        # Fit the model
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, X):
        starttime = time.time()
        # Normalize the features using training data statistics
        X_mean, X_std = X.mean(), X.std()
        X = (X - X_mean) / X_std
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Denormalize the predictions using training data statistics
        y_mean, y_std = train.iloc[:, -1].values.mean(), train.iloc[:, -1].values.std()
        predictions = (predictions * y_std) + y_mean
        endtime= time.time()
        pt = endtime - starttime
        return predictions , pt

# Create a sample DataFrame
data = np.loadtxt("airfoil_self_noise.dat")
cols = ["frequency", "angleofattack", "chordlength", "freestreamvelocity", "suctionsidedisplacement", "soundpressure"]
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
lr.fit(X, y)

# Make predictions
predictions = lr.predict(X_test)

# Calculate MSE loss
mse_loss = np.mean((predictions[0] - y_test) ** 2)

# print("Predictions:", predictions)
print("MSE Loss:", mse_loss)
print("time to predict" ,predictions[1] )
