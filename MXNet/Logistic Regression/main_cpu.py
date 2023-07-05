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
data_ctx = mx.cpu()
model_ctx = mx.cpu()

# %%
# setting the profiler for measuring the execution time and memory usage
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True,aggregate_stats=True,continuous_dump=False, filename='log_reg_gpu_profile.json')

# %% [markdown]
# Reading and pre-processing of data

# %%
# load the dataset
df = pd.read_csv('magic_gamma_telescope.csv', header=None)

# extract the features and labels
X = df.iloc[:, :-1].values.astype(np.float32)
y_labels = df.iloc[:, -1].values

# encode the string class labels as integers
y_labels[y_labels == 'g'] = 0
y_labels[y_labels == 'h'] = 1
y = y_labels.astype(np.int32)

# convert the features and labels to mxnet ndarrays
X = nd.array(X, ctx=data_ctx)
y = nd.array(y, ctx=data_ctx)
y = y.reshape((-1, 1))

# %%
print(X.shape)
print(y.shape)

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
weights = nd.random_normal(shape=(X.shape[1], 1), ctx=model_ctx)
bias = 0
num_samples, num_features = X.shape

# %%
data_set = gluon.data.ArrayDataset(X, y)
data_loader = gluon.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

# %%
# Defining the neural network
net = nn.Sequential()
# Adding the output layer with only one output, with the sigmoid activation function
with net.name_scope():
    net.add(nn.Dense(units=1, activation='sigmoid'))
# collect_params() will initialize the weights and biases for the neural network
net.collect_params().initialize(mx.init.Zero(), ctx=model_ctx)

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


