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
from mxnet import profiler
import re

# %%
data_ctx = mx.gpu()
model_ctx = mx.gpu()
# load the dataset
data = pd.read_csv('custom_2017_2020.csv')
# convert to numpy array
data = data.to_numpy()

# %%
# setting the profiler for measuring the execution time and memory usage
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='lin_reg_gd_gpu_profile.json')

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
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
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
# waiting for all operations to end, then stopping the profiler
mx.nd.waitall()
end = time.time()
profiler.set_state('stop')

# %%
results = profiler.dumps()

# %%
result = results
result = result.split('\n')

# %%
# splitting the result into a list of lists
for i in range(len(result)):
    result[i] = result[i].split()

# %%
# extracting the maximum gpu and cpu memory usage and the total execution time
max_gpu_use = 0
max_cpu_use = 0
total_execution_time = 0
# traversing over the lists and trying to find the maximum gpu and cpu memory usage and the total execution time
for i in result:
    if (len(i)>=1 and i[0]=='Memory:'):
        if (i[1]=='gpu/0'):
            max_gpu_use = float(i[-2])
        elif (i[1]=='cpu/0'):
            max_cpu_use = float(i[-2])
        else: continue
    # if the length of the list 6 and the second to sixth elements are numbers, then it is a time entry
    else:
        if (len(i)>=6):
            # if it is a valid time entry, then add it to the total execution time
            if (re.match(r'^-?\d+(?:\.\d+)$', i[-4]) is not None):
                total_execution_time += float(i[-4])

if (total_execution_time==0):
    total_execution_time = (end - start)*1000

# %%
print(f"Maximum GPU memory usage: {max_gpu_use} KB")
print(f"Maximum CPU memory usage: {max_cpu_use} KB")
print(f"Total execution time: {total_execution_time} milli seconds (ms)")


