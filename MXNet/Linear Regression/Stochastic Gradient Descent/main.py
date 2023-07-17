# %% [markdown]
# Importing required libraries and setting the profiler

# %%
# importing the libraries
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
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
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='lin_reg_sgd_profile.json')

# %% [markdown]
# Importing the dataset and preprocessing it 

# %%
# read data with pandas
data = pd.read_csv('train_data.csv')
# convert to numpy array
data = data.to_numpy()
test_data = pd.read_csv('test_data.csv')
# convert to numpy array
test_data = test_data.to_numpy()

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
batch_size = 1000

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
# Preparing the dataset for the neural net

# %%
# combining the features and labels into a single dataset
train_gdata = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# %% [markdown]
# Setting up the neural net for training
# 

# %%
num_epochs = 3
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

# %% [markdown]
# Training the data

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# training loop for the linear regression model
for epoch in range(num_epochs):
    for X, y in train_gdata:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(X_train), y_train)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

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

# %%
print(f"Maximum training GPU memory usage: {max_gpu_use} KB")
print(f"Maximum training CPU memory usage: {max_cpu_use} KB")
print(f"Total training time: {total_execution_time} milli seconds (ms)")

# %% [markdown]
# Reseting the profiler for the prediction phase

# %%
# set the profiler to default
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = False, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='lin_reg_sgd_profile.json')

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# predicting the training set
train_predictions = net(X_train)
# predicting the test set
test_predictions = net(X_test)

# %%
mse_train = nd.mean((train_predictions - y_train) ** 2)
mse_test = nd.mean((test_predictions - y_test) ** 2)

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
print(f"Mean Squared Error (MSE) of training set: {mse_train.asscalar()}")
print(f"Mean Squared Error (MSE) of test set: {mse_test.asscalar()}")


