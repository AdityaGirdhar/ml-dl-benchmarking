import numpy as np
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

def mse(t1, t2):
    diff = t1 - t2
    return np.sum(diff * diff) / len(diff)


num_of_epochs = 1000
learning_rate = 0.001
input_size = x_data.shape[1]  # Number of features in the input data
no_of_samples = x_data.shape[0]

seed_value = 42
np.random.seed(seed_value)
w = np.random.randn(input_size, 1)
b = np.random.randn(1)
derivative_weights = np.zeros((input_size, 1))
derivative_bias = 0

preds = model(x_data)
loss = mse(preds, y_data)

start_time = time.time()


for epoch in range(num_of_epochs):
    preds = model(x_data)
    loss = mse(preds, y_data)
    difference = y_data - preds
    derivative_weights = np.dot(x_data.T, difference) / no_of_samples
    derivative_bias = np.sum(difference) / no_of_samples
    w = w + learning_rate * derivative_weights
    b = b + learning_rate * derivative_bias
    print('epoch {} loss {}'.format(epoch, loss))

end_time = time.time()



# elapsed_time = end_time - start_time

# # Print the elapsed time
# print("Time in numpy: ", elapsed_time, " seconds")

# # Making predictions on training data
# train_preds = model(x_data)
# train_acc = mse(y_data, train_preds)
# print("Training Accuracy of numpy: {}".format(train_acc))

# # Making predictions on testing data
# test_preds = model(x_data_test)
# test_acc = mse(y_data_test, test_preds)
# print("Testing Accuracy of numpy: {} ".format(test_acc))