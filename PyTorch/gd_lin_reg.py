import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import pandas as pd

device = torch.device("cpu")
torch.backends.cudnn.enabled = False

data = pd.read_csv('custom_2017_2020.csv')
# convert to numpy array
data = data.to_numpy()

# standardize the features
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std

d_tensor = torch.tensor(data, device=device)
x_data = torch.tensor(data[:, :-1], device=device)
y = torch.tensor(data[:, -1], device=device)
y_data = torch.reshape(y, (len(y),1))

# print('x_data shape', x_data.shape)
# print('y_data shape', y_data.shape)
# print(x_data)
# print(y_data)

# no_of_data_points = x_data.shape[0]
# no_of_features = x_data.shape[1]
# no_of_epochs = 1000

# # weights initialisation
# weights = np.random.normal(0, 1, (no_of_features, 1))
# # bias initialisation
# bias = np.random.normal(0, 1, (1, 1))

# learning_rate = 0.0001
# derivative_weights = np.zeros((no_of_features, 1))
# derivative_bias = 0

# weights = torch.tensor(weights)
# bias = torch.tensor(bias)
# derivative_weights = torch.tensor(derivative_weights)

# for epoch in range(no_of_epochs):
#     y_pred = torch.matmul(x_data, weights) + bias
#     difference = y_data - y_pred
#     derivative_weights = torch.matmul(x_data.t(), difference) / no_of_data_points
#     derivative_bias = torch.sum(difference) / no_of_data_points
#     # updating the weights and bias
#     weights = weights + learning_rate * derivative_weights
#     bias = bias + learning_rate * derivative_bias
#     mse_loss = torch.mean(difference ** 2)
#     print('epoch',epoch,'loss',mse_loss.item())


def model(x):
    #w = w.to(x.dtype)
    return torch.matmul(x, w.t()) + b

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

num_of_epochs = 1000
learning_rate = 0.001
input_size = x_data.shape[1]  # Number of features in the input data
no_of_samples = x_data.shape[0]

# Initialize the weight tensor with appropriate shape
w = torch.randn(1, input_size, requires_grad=True, dtype=torch.float64, device=device)
b = torch.randn(1, requires_grad=True,  dtype=torch.float64, device=device)
# print(w.shape)
# print(b.shape)
preds = model(x_data)
# print(preds.shape)
# print(y_data.shape)
loss = mse(preds, y_data)
#print(loss.backward)
# print("Device for x_data:", x_data.device)
# print("Device for y_data:", y_data.device)
# print("Device for w:", w.device)
# print("Device for b:", b.device)

start_time = time.time()

for epoch in range(num_of_epochs):
    preds = model(x_data)
    loss = mse(preds, y_data)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()
    #print('epoch {}, loss {}'.format(epoch, loss))
end_time = time.time()

elapsed_time = end_time - start_time

# # Print the elapsed time
print("Time in pytorch: ", elapsed_time, " seconds")
