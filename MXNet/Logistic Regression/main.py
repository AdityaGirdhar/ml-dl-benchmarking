# %% [markdown]
# Importing all the libraries and packages required

# %%
# importing packages and libraries
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import profiler
import time
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
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True,aggregate_stats=True,continuous_dump=False, filename='log_reg_sgd_profile.json')

# %% [markdown]
# Reading and pre-processing of data

# %%
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# %%
X_train = nd.array(train_data[:,1:], ctx=data_ctx)
y_train = nd.array(train_data[:,0], ctx=data_ctx)
X_test = nd.array(test_data[:,1:], ctx=data_ctx)
y_test = nd.array(test_data[:,0], ctx=data_ctx)
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

# %% [markdown]
# Initialising the model for training

# %%
def sigmoid(z):
    return 1 / (1 + nd.exp(-z))

# %%
# initialize the parameters
learning_rate = 0.1
num_of_epochs = 10
batch_size = 10
weights = nd.random_normal(shape=(X_train.shape[1], 1), ctx=model_ctx)
bias = 0
num_samples, num_features = X_train.shape

# %%
data_set = gluon.data.ArrayDataset(X_train, y_train)
data_loader = gluon.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

# %%
# Defining the neural network
net = nn.Sequential()
# Adding the output layer with only one output, with the sigmoid activation function
with net.name_scope():
    net.add(nn.Dense(units=1, activation='sigmoid'))
# collect_params() will initialize the weights and biases for the neural network
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# %%
# defining the loss function and the optimizer
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(params = net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': learning_rate})

# %% [markdown]
# Training the model

# %%
def training_function():
    cumulative_loss = 0
    # for each epoch, iterating over the dataset in batches
    for i, (data, label) in enumerate(data_loader):
        # for the forward pass
        with autograd.record():
            output = net(data)      # output is the predicted value from the neural network
            L = loss(output, label) # L will store the loss between the predicted value and the actual value
        L.backward()                # for the backward pass
        trainer.step(batch_size)    # updating the weights and biases
        cumulative_loss += nd.sum(L).asscalar()

# %%
# running one epoch before profiling
training_function()

# %%
mx.nd.waitall() 

# starting the profiler
start = time.time()
profiler.set_state('run')

# %%
for epoch in range(num_of_epochs):
    training_function()

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
# Parsing the profiler

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

# %% [markdown]
# Reseting the profiler for the prediction step

# %%
# set the profiler to default
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = False, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='log_reg_sgd_profile.json')

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# computing the predicted values 
predicted_train = net(X_train)
predicted_test = net(X_test)
predicted_train[predicted_train >= 0.5] = 1
predicted_train[predicted_train < 0.5] = 0
predicted_test[predicted_test >= 0.5] = 1
predicted_test[predicted_test < 0.5] = 0

# %%
# waiting for all operations to end, then stopping the profiler
mx.nd.waitall()
end = time.time()
profiler.set_state('stop')

# %%
# computing the accuracy
train_accuracy = nd.mean(predicted_train == y_train)
test_accuracy = nd.mean(predicted_test == y_test)

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
print(f"Training accuracy: {train_accuracy.asscalar()*100}%")
print(f"Testing accuracy: {test_accuracy.asscalar()*100}%")


