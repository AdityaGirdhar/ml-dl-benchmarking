import numpy as np
import pandas as pd
import time

data = pd.read_csv('custom_2017_2020.csv')
# convert to numpy array
data = data.to_numpy()

# standardize the features
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std

features = np.array(data[:, :-1])
labels = np.array(data[:, -1])

X_train = features
y_train = labels
y_train = y_train.reshape((len(y_train), 1))

no_of_data_points = X_train.shape[0]
no_of_features = X_train.shape[1]
no_of_epochs = 1000

weights = np.random.normal(0, 1, (no_of_features, 1))
# bias initialisation
bias = np.random.normal(0, 1, (1, 1))

# %%
# setting the hyperparameters
learning_rate = 0.00001
derivative_weights = np.zeros((no_of_features, 1))
derivative_bias = 0

start = time.time()

# %%
for epoch in range(no_of_epochs):
    y_pred = np.dot(X_train, weights) + bias
    difference = y_train - y_pred
    derivative_weights = np.dot(X_train.T, difference) / no_of_data_points
    derivative_bias = np.sum(difference) / no_of_data_points
    # updating the weights and bias
    weights = weights + learning_rate * derivative_weights
    bias = bias + learning_rate * derivative_bias
    #mse_loss = np.mean(difference**2)
    #print("Epoch:", epoch, "MSE Loss:", mse_loss)

# %%
end = time.time()

# %%
print(f"Time in numpy: {end - start} seconds")