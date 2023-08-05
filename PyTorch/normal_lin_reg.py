#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import time

data = np.genfromtxt('../custom_2017_2020.csv', delimiter=',')

X = data[1:] # Removing first row containing feature labels
y = X[:, -1]

X = np.delete(X, -1, axis=1)
oneX = np.ones((len(X), 1))
X = np.concatenate([oneX, X], axis=1)

tensor_data = torch.tensor(X)
tensor_y = torch.tensor(y)

print('Data imported, fitting model.')

# Model training starts
start_time = time.time()
dtd = torch.matmul(tensor_data.t(), tensor_data)
theta = torch.matmul(torch.matmul(torch.inverse(dtd), tensor_data.t()), tensor_y)
end_time = time.time()
# Model training ends

elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time: ", elapsed_time, " seconds")
