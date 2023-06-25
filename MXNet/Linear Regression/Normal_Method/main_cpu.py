# %% [markdown]
# ### Linear Regression
# 1. Ordinary Least Squares Method: 
# In this method, we find the regression coefficient weights that minimize the sum of the squared residuals.
# 
# Formula:  $$ weights = (X^T \cdot X)^{-1} \cdot X^T \cdot y$$
# 
# To find the predicted values, we multiply the feature matrix X with the weights vector.
# Formula: $$ y_{pred} = X \cdot weights $$
# 
# Mean Squared Error: $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{true})^2 $$

# %%
# importing the libraries
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import LinearRegression as lr
import time
import pandas as pd

# %%
# read data from DAT file in NDArray format
data_ctx = mx.cpu()
model_ctx = mx.cpu()
# read data with pandas
data = pd.read_csv('custom_2017_2020.csv')
# convert to numpy array
data = data.to_numpy()

# %% [markdown]
# Pre-processing step

# %%
# standardize the features
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std
# convert to NDArray
data = nd.array(data, ctx=data_ctx)

# %%
# splitting into features and labels
features = nd.array(data[:, :-1], ctx=data_ctx)
labels = nd.array(data[:, -1], ctx=data_ctx)

# %%
X_train = features
y_train = labels
y_train = y_train.reshape((len(y_train), 1))

# %%
# printing the shapes of the training set to check dimensions
print(X_train.shape)
print(y_train.shape)

# %% [markdown]
# Training the model

# %%
start = time.time()

# %%
# training the model using OLS method
model = lr.LinearRegression()
# note the time before starting the training of the data
weights = model.OLS_fit(X_train, y_train)

# %%
end = time.time()

# %% [markdown]
# Printing the result

# %%
print(f"Time taken to train the model using GPU: {end - start} seconds")


