# %%
# importing packages and libraries
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import pandas as pd
import time
data_ctx = mx.cpu()
model_ctx = mx.cpu()

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
end = time.time()

# %%
print(f"Time taken to train the Naive Bayes classifier: {end - start}")


