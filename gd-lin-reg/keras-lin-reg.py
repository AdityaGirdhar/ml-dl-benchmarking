import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import tensorflow as tf

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


x_data = train_data[:, :-1]
y = train_data[:, -1]
y_data = y.reshape(len(y),1)

x_data_test = test_data[:, :-1]
y_data_test = test_data[:, -1]
y_data_test = y_data_test.reshape((len(y_data_test), 1))


num_of_epochs = 1000
learning_rate = 0.001
input_size = x_data.shape[1] 

model = Sequential()
model.add(Dense(1, input_shape=(input_size,), activation=None))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

np.random.seed(42)
w_np = np.random.randn(input_size, 1)
b_np = np.random.randn(1)
train_preds = model.predict(x_data)
loss = mean_squared_error(y_data, train_preds)

for epoch in range(num_of_epochs):
    train_preds = model.predict(x_data)

    # Calculate gradients of the mean squared error loss with respect to the weights
    with tf.GradientTape() as tape:
        loss = mean_squared_error(y_data, train_preds)
    gradients = tape.gradient(loss, model.trainable_variables)

    # Update weights
    for param, grad in zip(model.trainable_variables, gradients):
        updated_param = param.numpy() - learning_rate * grad.numpy()
        param.assign(updated_param)
    
    train_mse_loss = np.mean((y_data - train_preds) ** 2)
    

    print(f"Epoch {epoch+1} - Training MSE Loss: {train_mse_loss}")