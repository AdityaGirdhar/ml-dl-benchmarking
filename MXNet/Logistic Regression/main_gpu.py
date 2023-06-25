# %% [markdown]
# Importing all the libraries and packages required

# %%
# importing packages and libraries
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import pandas as pd
import time
data_ctx = mx.gpu()
model_ctx = mx.gpu()

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
print(num_samples, num_features)

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
start = time.time()

# %%
# training the model over the number of epochs
for epoch in range(num_of_epochs):
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
end = time.time()

# %%
print(f"Time taken to run the model: {end - start} seconds")


