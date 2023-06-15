import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv('airfoil_self_noise.dat', sep='\t', header=None)
x_data = Variable(torch.Tensor(data.iloc[:, :-1].values))
y_data = Variable(torch.Tensor(data.iloc[:, -1].values).view(-1, 1))

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(5, 1)  # Five input features and one output

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Create the model
our_model = LinearRegressionModel()

# Training loop
epochs = 500
lr = 0.01

for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    pred_y = our_model(x_data)

    # Compute the difference between predicted y and actual y
    loss = pred_y - y_data

    # Update the weights directly without using optimization algorithm
    our_model.linear.weight.data -= lr * torch.matmul(x_data.t(), loss).squeeze() / len(x_data)
    our_model.linear.bias.data -= lr * loss.sum().item() / len(x_data)


    print('epoch {}, loss {}'.format(epoch, loss.sum()))

# Test the model on a new input
new_var = Variable(torch.Tensor([[800, 0.3, 0.02, 30, 120]]))
pred_y = our_model(new_var)
print("Predicted sound pressure level:", pred_y.item())
