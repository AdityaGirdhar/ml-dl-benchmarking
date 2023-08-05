import pandas as pd

# Read the CSV file skipping the first row to exclude the header
data = pd.read_csv('custom_2017_2020.csv')

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(data)
print(data.shape)


# Calculate the index to split the data
split_index = int(0.8 * len(data))
# print(split_index)

# Split the data into training and testing sets
train_data = data.loc[:split_index - 1]
test_data = data.loc[split_index:]


# print(train_data)
# print(test_data)
# print(train_data.shape)
# print(test_data.shape)

# Save the training set as CSV
train_data.to_csv('train_data.csv', index=False)

# # Save the testing set as CSV
test_data.to_csv('test_data.csv', index=False)
