import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import pandas as pd
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cpu")
torch.backends.cudnn.enabled = False

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


d_tensor = torch.tensor(train_data, device = device)
x_data = torch.tensor(train_data[:, :-1], device = device)
y = torch.tensor(train_data[:, -1], device = device)
y_data = torch.reshape(y, (len(y),1))

d_tensor_test = torch.tensor(test_data, device = device)
x_data_test = torch.tensor(test_data[:, :-1], device = device)
y_test = torch.tensor(test_data[:, -1], device = device)
y_data_test = torch.reshape(y_test, (len(y_test),1))

# print('x_data shape', x_data.shape)
# print('y_data shape', y_data.shape)
# print(x_data)
# print(y_data)

def model(x):
    return torch.matmul(x, w) + b

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

num_of_epochs = 1000
learning_rate = 0.001
input_size = x_data.shape[1]  # Number of features in the input data
no_of_samples = x_data.shape[0]

# Initialize the weight tensor with appropriate shape
# w = torch.randn(input_size, 1, requires_grad=True, dtype=torch.float64, device=device)
# b = torch.randn(1, requires_grad=True,  dtype=torch.float64, device=device)

seed_value = 42
np.random.seed(seed_value)
w_np = np.random.randn(input_size, 1)
b_np = np.random.randn(1)

w = torch.tensor(w_np, requires_grad=True, dtype=torch.float64, device=device)
b = torch.tensor(b_np, requires_grad=True, dtype=torch.float64, device=device)


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

end_time = time.time()

print('loss {}'.format(loss))

elapsed_time = end_time - start_time

# # Print the elapsed time
print("Time in pytorch: ", elapsed_time, " seconds")


# Making predictions on training data
train_preds = model(x_data)
train_acc = mse(y_data, train_preds)
print("Training Accuracy of pytorch: {}".format(train_acc))

# Making predictions on testing data
test_preds = model(x_data_test)
test_acc = mse(y_data_test, test_preds)
print("Testing Accuracy of pytorch: {} ".format(test_acc))