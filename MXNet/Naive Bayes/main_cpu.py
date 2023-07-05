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
profiler.set_config(profile_all=False,profile_symbolic = False, profile_imperative = False,profile_memory = True, profile_api = True, aggregate_stats=True,continuous_dump=False, filename='naive_bayes_gpu_profile.json')

# %%
# load the dataset
df = pd.read_csv('magic_gamma_telescope.csv', header=None)

# %% [markdown]
# Pre-processing step

# %%
# convert into numpy array
dataset = df.values
dataset[:, -1] = [1 if x == 'g' else 0 for x in dataset[:, -1]]

# %%
# splitting the data into unique classes's data
X_0 = nd.array(dataset[dataset[:, -1] == 0][:, :-1], ctx = data_ctx)
X_1 = nd.array(dataset[dataset[:, -1] == 1][:, :-1], ctx = data_ctx)
# removing the target variable from the dataset
dataset = dataset[:, :-1]
# changing the datatype to float due to standard deviation computation issues
dataset = np.array(dataset, dtype=np.float64)

# %% [markdown]
# Training step

# %%
mx.nd.waitall() 

# starting the profiler
profiler.set_state('run')
start = time.time()

# %%
start = time.time()

# %%
# Class Prior Probabilities
# computing the prior probabilities of the two classes
prior_prob_0 = X_0.shape[0] / dataset.shape[0]
prior_prob_1 = X_1.shape[0] / dataset.shape[0]

# %%
# Class specific statistics
# computing the mean and the standard deviations of the features of the two classes
mean_0 = nd.mean(X_0, axis = 0)
mean_1 = nd.mean(X_1, axis = 0)
std_0 = nd.array(np.std(X_0.asnumpy(), axis = 0), ctx = data_ctx)
std_1 = nd.array(np.std(X_1.asnumpy(), axis = 0), ctx = data_ctx)

# %%
# Overall statistics computation
# computing the mean and the standard deviations of the features 
mean_total = nd.array(np.mean(dataset, axis = 0))
std_total = nd.array(np.std(dataset, axis = 0))

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


