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

# %% [markdown]
# Importing required libraries and setting the profiler

# %%
# importing the libraries
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import LinearRegression_GPU as lr_gpu
import time
import pandas as pd
from mxnet import profiler
import re

# %%
# setting the context for the program 
mx.test_utils.list_gpus()
if mx.context.num_gpus() > 0:
    data_ctx = mx.gpu()
    model_ctx = mx.gpu()
else:
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

# %%
# setting the profiler for measuring the execution time and memory usage
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='lin_reg_nf_gpu_profile.json')

# %% [markdown]
# Importing the dataset

# %%
# read data with pandas
data = pd.read_csv('train_data.csv')
# convert to numpy array
data = data.to_numpy()
test_data = pd.read_csv('test_data.csv')
# convert to numpy array
test_data = test_data.to_numpy()

# %% [markdown]
# Pre-processing step - Standardizing the data

# %%
# standardize the features
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std
# convert to NDArray
data = nd.array(data, ctx=data_ctx)

# standardize the features
test_data = (test_data - mean) / std
# convert to NDArray
test_data = nd.array(test_data, ctx=data_ctx)

# %% [markdown]
# Splitting the dataset into train and test sets

# %%
# splitting into features and labels
features = nd.array(data[:, :-1], ctx=data_ctx)
labels = nd.array(data[:, -1], ctx=data_ctx)

# %%
X_train = features
y_train = labels
y_train = y_train.reshape((len(y_train), 1))

# %%
X_test = nd.array(test_data[:, :-1], ctx=data_ctx)
y_test = nd.array(test_data[:, -1], ctx=data_ctx)
y_test = y_test.reshape((len(y_test), 1))

# %%
# printing the shapes of the training set to check dimensions
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# Training the model

# %% [markdown]
# Marking the profiler for the training step

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# training the model using OLS method
model = lr_gpu.LinearRegression()
# note the time before starting the training of the data
weights = model.OLS_fit(X_train, y_train)

# %% [markdown]
# Ending the profiler for the training step

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

# %% [markdown]
# Parsing the profiler data

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

# %% [markdown]
# Printing the result

# %%
print(f"Maximum training GPU memory usage: {max_gpu_use} KB")
print(f"Maximum training CPU memory usage: {max_cpu_use} KB")
print(f"Total training time: {total_execution_time} milli seconds (ms)")

# %% [markdown]
# Reseting the profiler for the prediction step

# %%
# set the profiler to default
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = False, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='lin_reg_nf_gpu_profile.json')

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# predicting the values
predictions = model.OLS_predict(X_test, weights)
# prediction on the training set
predictions_train = model.OLS_predict(X_train, weights)

# %%
# waiting for all operations to end, then stopping the profiler
mx.nd.waitall()
end = time.time()
profiler.set_state('stop')

# %%
# calculating the mean squared error
mse = nd.mean(((predictions - y_test) ** 2))
# calculating the mean squared error
mse_train = nd.mean(((predictions_train - y_train) ** 2))

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
# parsing the profiler data
total_execution_time = 0
for i in result:
    if (len(i)>=6):
        if (re.match(r'^-?\d+(?:\.\d+)$', i[-4]) is not None):
            total_execution_time += float(i[-4])

if (total_execution_time==0):
    total_execution_time = (end - start)*1000

# %%
print(f"Total prediction/testing time: {total_execution_time} milli seconds (ms)")
print(f"Mean Squared Error on training set: {mse_train.asscalar()}")
print(f"Mean Squared Error on test set: {mse.asscalar()}")


