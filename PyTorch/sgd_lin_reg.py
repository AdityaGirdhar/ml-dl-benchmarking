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

d_tensor = torch.tensor(data, device=device, dtype=torch.float32)
x_data = torch.tensor(data[:, :-1], device=device, dtype=torch.float32)
y = torch.tensor(data[:, -1], device=device, dtype=torch.float32)
y_data = torch.reshape(y, (len(y),1))



class LinearRegressionModel(torch.nn.Module):
    def __init__(self, device):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(12, 1).to(device)  # One in and one out, with device specified

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Create the model with device parameter
our_model = LinearRegressionModel(device)

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.00000000001)
pred_y = our_model(x_data)
loss = criterion(pred_y, y_data)

start_time = time.time()

for epoch in range(1000):

	# Forward pass: Compute predicted y by passing
	# x to the model
	pred_y = our_model(x_data)

	# Compute and print loss
	loss = criterion(pred_y, y_data)

	# Zero gradients, perform a backward pass,
	# and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	#print('epoch {}, loss {}'.format(epoch, loss.item()))

end_time = time.time()

elapsed_time = end_time - start_time
#Print the elapsed time
print("Elapsed time: ", elapsed_time, " seconds")

# new_var = Variable(torch.Tensor([[4.0]]))
# pred_y = our_model(new_var)
# print("predict (after training)", 4, our_model(new_var).item())
