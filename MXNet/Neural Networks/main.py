# %%
from mxnet import nd
from mxnet.gluon import nn
import pandas as pd
import mxnet as mx
from mxnet import profiler
import re
import time
# setting the context for the program 
mx.test_utils.list_gpus()
if mx.context.num_gpus() > 0:
    data_ctx = mx.gpu()
    model_ctx = mx.gpu()
else:
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()
mx.random.seed(42, ctx=model_ctx)

# %%
# setting the profiler for measuring the execution time and memory usage
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='neural_net_gpu_profile.json')

# %%
# loading the dataset using mx.test_utils
mnist = mx.test_utils.get_mnist()
batch_size = 100
learning_rate = 0.1

# Convert training and validation data to NDArray format
train_data_array = mx.nd.array(mnist['train_data'], ctx=data_ctx)
train_label_array = mx.nd.array(mnist['train_label'], ctx=data_ctx)
test_data_array = mx.nd.array(mnist['test_data'], ctx=data_ctx)
test_label_array = mx.nd.array(mnist['test_label'], ctx=data_ctx)

# Create an iterator with combined data
combined_data_iter = mx.io.NDArrayIter(train_data_array, train_label_array, batch_size, shuffle=True)
val_data_iter = mx.io.NDArrayIter(test_data_array, test_label_array, batch_size, shuffle=True)

# %% [markdown]
# Neural network construction

# %%
# defining a neural net
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(units=200, activation='relu', use_bias=True, dtype='float32', in_units=784))
    net.add(nn.Dense(units=100, activation='relu', use_bias=True, dtype='float32', in_units=200))
    net.add(nn.Dense(units=10, activation=None, use_bias=True, dtype='float32', in_units=100))

# %%
# Initializing the parameters for the neural net
net.initialize(mx.initializer.Uniform(), ctx=model_ctx)

# %%
# defining the trainer with SGD optimizer
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

# %%
# setting the number of epochs and metric
num_of_epochs = 10
metric = mx.metric.Accuracy()
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

# %% [markdown]
# Training loop

# %%
def training_function():
    combined_data_iter.reset()
    for batch in combined_data_iter:
        data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=[model_ctx], batch_axis=0)
        label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=[model_ctx], batch_axis=0)
        outputs = []
        with mx.autograd.record():
            for x, y in zip(data, label):
                z = net(x)
                loss_value = loss(z, y)
                loss_value.backward()
                outputs.append(z)
        trainer.step(batch.data[0].shape[0])
        metric.update(label, outputs)
    metric.reset()

# %%
# running the iteration once before starting the profiler
training_function()

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

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

# %%
# set the profiler to default
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = False, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='neural_net_gpu_profile.json')

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
# predictions on the training data
combined_data_iter.reset()
for batch in combined_data_iter:
    data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list = [model_ctx], batch_axis=0)
    label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list = [model_ctx], batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(net(x))
    metric.update(label, outputs)
combined_accuracy = metric.get()
# predictions on the validation data
val_data_iter.reset()
for batch in val_data_iter:
    data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list = [model_ctx], batch_axis=0)
    label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list = [model_ctx], batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(net(x))
    metric.update(label, outputs)
val_accuracy = metric.get()

# %%
# waiting for all operations to end, then stopping the profiler
mx.nd.waitall()
end = time.time()
profiler.set_state('stop')

# %%
# evaluating the performance on the predictions
combined_accuracy = combined_accuracy[1]
val_accuracy = val_accuracy[1]

# %%
results = profiler.dumps()
result = results
result = result.split('\n')
# splitting the result into a list of lists
for i in range(len(result)):
    result[i] = result[i].split() 
# parsing the profiler data
total_execution_time = 0
for i in result:
    if (len(i)>=6):
        if (re.match(r'^-?\d+(?:\.\d+)$', i[-4]) is not None):
            total_execution_time += float(i[-4])

if (total_execution_time==0):
    total_execution_time = (end - start)*1000

# %%
print(f"Total execution time: {total_execution_time} milli seconds (ms)")
print(f"Training accuracy: {combined_accuracy*100} %")
print(f"Validation accuracy: {val_accuracy*100} %")


