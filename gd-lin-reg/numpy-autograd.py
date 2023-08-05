import autograd.numpy as np
from autograd import grad
import pandas as pd
import time

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

x_data = np.array(train_data[:, :-1])
y_data = np.array(train_data[:, -1])
y_data = y_data.reshape((len(y_data), 1))

x_data_test = np.array(test_data[:, :-1])
y_data_test = np.array(test_data[:, -1])
y_data_test = y_data_test.reshape((len(y_data_test), 1))

def model(x):
    return np.dot(x, w) + b

def loss_test(w,b):
  y_hat = np.dot(x_data_test,w) + b
  return np.sum((y_hat - y_data)**2) / len(x_data_test)

def loss(w,b):
  y_hat = np.dot(x_data,w) + b
  return np.sum((y_hat - y_data_test)**2) / len(x_data)

num_of_epochs = 1000
eta = 0.001
input_size = x_data.shape[1]  # Number of features in the input data
no_of_samples = x_data.shape[0]

seed_value = 42
np.random.seed(seed_value)
w = np.random.randn(input_size, 1)
b = np.random.randn(1)

l = loss(w, b)
grad_loss_w = grad(loss,0)
grad_loss_b = grad(loss,1)

start_time = time.time()
for i in range(num_of_epochs):
    grad_w, grad_b  = grad_loss_w(w,b),grad_loss_b(w,b)
    w = w - eta * grad_w
    b = b - eta * grad_b
    # if (i % 100 == 0):
    #   print("Epoch:",i, "mse=", loss(w,b))
end_time = time.time()

print(loss(w,b))
elapsed_time = end_time - start_time
print("Time in numpy: ", elapsed_time, " seconds")

# Making predictions on testing data

test_acc = loss_test(w,b)
print("Testing Accuracy of numpy: {} ".format(test_acc))