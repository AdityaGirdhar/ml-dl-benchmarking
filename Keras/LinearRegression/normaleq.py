import numpy as np
import pandas as pd
import time

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        starttime = time.time()
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Compute the weights using the normal equation
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        endtime = time.time()
        pt = endtime - starttime
        return pt
    def predict(self, X):
        starttime = time.time()
        # Add a column of ones to X for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Make predictions
        predictions = X.dot(self.weights)
        
        endtime = time.time()
        pt = endtime - starttime
        return predictions, pt


# Create a sample DataFrame
data = np.loadtxt(r'Keras\LinearRegression\dataset\custom_2017_2020.dat',delimiter=',')
cols = ["exp_imp", "Year", "month", "ym", "Country", "Custom", "hs2", "hs4", "hs6", "hs9", "Q1", "Q2", "Value"]
df = pd.DataFrame(data, columns=cols)
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
tt= lr.fit(X, y)
print("time to fit the model", tt)
# Make predictions
predictions = lr.predict(X_test)

# Calculate MSE loss
mse_loss = np.mean((predictions[0] - y_test) ** 2)

print("MSE Loss:", mse_loss)
print("Time to predict:", predictions[1])
