# %% [markdown]
# ### Linear Regression
# #### Linear Regression with Gradient Descent 
# In this method, we find the regression coefficient weights that minimize the sum of the squared residuals.
# The formulation of the loss function is given as-
# Formula: $$ L = \frac{1}{2n} \sum_{i=1}^{n} (y_{pred} - y_{true})^2 $$
# The gradient of the loss function is given as-
# Formula: $$ \frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{true}) \cdot x $$
# The weights are updated as-
# Formula: $$ w = w - \alpha \cdot \frac{\partial L}{\partial w} $$
# where $\alpha$ is the learning rate.

# %%
# importing packages and libraries
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import pandas as pd
import time 

# %%
data_ctx = mx.cpu()
model_ctx = mx.cpu()
# load the dataset
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
# Model parameters initialization

# %%
# declaring the parameters and important variables
no_of_data_points = X_train.shape[0]
no_of_features = X_train.shape[1]
no_of_epochs = 1000

# %%
# weights initialisation
weights = np.random.normal(0, 1, (no_of_features, 1))
# bias initialisation
bias = np.random.normal(0, 1, (1, 1))

# %%
# setting the hyperparameters
learning_rate = 0.0001
derivative_weights = np.zeros((no_of_features, 1))
derivative_bias = 0

# %%
# converting to NDArray
weights = nd.array(weights, ctx=model_ctx)
bias = nd.array(bias, ctx=model_ctx)
derivative_weights = nd.array(derivative_weights, ctx=model_ctx)

# %% [markdown]
# Training the model

# %%
start = time.time()

# %%
for epoch in range(no_of_epochs):
    y_pred = nd.dot(X_train, weights) + bias
    difference = y_train - y_pred
    derivative_weights = nd.dot(X_train.T, difference) / no_of_data_points
    derivative_bias = nd.sum(difference) / no_of_data_points
    # updating the weights and bias
    weights = weights + learning_rate * derivative_weights
    bias = bias + learning_rate * derivative_bias

# %%
end = time.time()

# %%
print(f"Time taken to train the model using gradient descent: {end - start} seconds")


