from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import pandas as pd
import time


data_ctx = mx.cpu()
model_ctx = mx.cpu()

# reading the data
train_data = pd.read_csv('../train_data.csv')
test_data = pd.read_csv('../test_data.csv')

# convert to numpy array
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# standardize the features
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean) / std

# mean_test = np.mean(test_data, axis=0)
# std_test = np.std(test_data, axis=0)
test_data = (test_data - mean) / std

d_tensor = nd.array(train_data, ctx = data_ctx)
x_data = nd.array(train_data[:, :-1], ctx = data_ctx)
y = nd.array(train_data[:, -1], ctx = data_ctx)
y_data = nd.reshape(y, (len(y),1))

d_tensor_test = nd.array(test_data, ctx = data_ctx)
x_data_test = nd.array(test_data[:, :-1], ctx = data_ctx)
y_test = nd.array(test_data[:, -1], ctx = data_ctx)
y_data_test = nd.reshape(y_test, (len(y_test),1))

# declaring the parameters and important variables
num_of_epochs = 1000
learning_rate = 0.001
no_of_data_points = x_data.shape[0]
no_of_features = x_data.shape[1]

def model(x):
    return nd.dot(x, weights) + bias

def mse(t1, t2):
    diff = t1 - t2
    return nd.sum(diff * diff) / diff.size


# weights initialisation
# weights = np.random.normal(0, 1, (no_of_features, 1))
# # bias initialisation
# bias = np.random.normal(0, 1, (1, 1))
# derivative_weights = np.zeros((no_of_features, 1))
# derivative_bias = 0

# # converting to NDArray
# weights = nd.array(weights, ctx=model_ctx)
# bias = nd.array(bias, ctx=model_ctx)
# derivative_weights = nd.array(derivative_weights, ctx=model_ctx)



# for epoch in range(num_of_epochs):
#     y_pred = model(x_data)
#     loss = mse(y_pred, y_data)
#     difference = y_data - y_pred
#     derivative_weights = nd.dot(x_data.T, difference) / no_of_data_points
#     derivative_bias = nd.sum(difference) / no_of_data_points
#     # updating the weights and bias
#     weights = weights + learning_rate * derivative_weights
#     bias = bias + learning_rate * derivative_bias

#     print('loss {}'.format(loss))

# weights = nd.random_normal(shape=(no_of_features, 1), ctx=model_ctx)
# bias = nd.random_normal(shape=(1,), ctx=model_ctx)
seed_value = 42
np.random.seed(seed_value)

weights_np = np.random.normal(size=(no_of_features, 1))
bias_np = np.random.normal(size=(1,))
weights = nd.array(weights_np, ctx=model_ctx)
bias = nd.array(bias_np, ctx=model_ctx)

weights.attach_grad()
bias.attach_grad()
y_pred = model(x_data)
loss = mse(y_pred, y_data)

# training loop
start_time = time.time()

for epoch in range(num_of_epochs):
    with autograd.record():
        y_pred = model(x_data)
        loss = mse(y_pred, y_data)
    loss.backward()

    # updating the weights and bias
    weights -= learning_rate * weights.grad
    bias -= learning_rate * bias.grad

    # clearing gradients for the next iteration
    weights.grad[:] = 0
    bias.grad[:] = 0
    
mx.nd.waitall()
end_time = time.time()
print('loss {}'.format(loss.asscalar()))

elapsed_time = end_time - start_time

# # Print the elapsed time
print("Time in mxnet: ", elapsed_time, " seconds")

train_preds = model(x_data)
train_acc = mse(y_data, train_preds)
print("Training Accuracy mxnet: {}".format(train_acc))

# Making predictions on testing data
test_preds = model(x_data_test)
test_acc = mse(y_data_test, test_preds)
print("Testing Accuracy mxnet: {} ".format(test_acc))