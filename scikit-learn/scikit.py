#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time

data = np.genfromtxt('../custom_2017_2020.csv', delimiter=',')

X = data[1:] # Removing first row containing feature labels
y = X[:, -1]

X = np.delete(X, -1, axis=1)
oneX = np.ones((len(X), 1))
X = np.concatenate([oneX, X], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
start_time = time.time()
model.fit(X, y)
end_time = time.time()

elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time: ", elapsed_time, " seconds")

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)

# # Print the mean squared error
# print("Mean Squared Error:", mse)