import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000, display_step=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.display_step = display_step
        self.W = None
        self.b = None

    def fit(self, X_train, y_train):
        # Normalize the features
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        # Convert the NumPy arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # Define variables for weights and bias
        self.W = tf.Variable(tf.zeros([X_train.shape[1], 1]))
        self.b = tf.Variable(tf.zeros([1]))

        # Define the linear regression model
        def linear_regression(X):
            return tf.matmul(X, self.W) + self.b

        # Define the mean squared error loss function
        def loss_fn(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))

        # Optimizer
        optimizer = tf.keras.optimizers.SGD(self.learning_rate)

        # Start the training loop
        start_time = datetime.now()
        for epoch in range(self.num_epochs):
            indices = np.random.choice(X_train.shape[0], size=512, replace=False)
            X_batch = tf.gather(X_train, indices)
            y_batch = tf.gather(y_train, indices)

            with tf.GradientTape() as tape:
                # Forward pass
                pred = linear_regression(X_batch)
                loss = loss_fn(y_batch, pred)

            # Calculate gradients
            gradients = tape.gradient(loss, [self.W, self.b])

            # Update weights and bias
            optimizer.apply_gradients(zip(gradients, [self.W, self.b]))

            if (epoch + 1) % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss), " time:" ,datetime.now()-start_time)

        end_time = datetime.now()
        execution_time = end_time - start_time
        print("Training completed!")
        print("Execution time:", execution_time)

    def predict(self, X):
        # Normalize the features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Convert the NumPy array to TensorFlow tensor
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        # Perform the prediction
        pred = tf.matmul(X, self.W) + self.b

        return pred.numpy()


# Load and preprocess the data
data = np.loadtxt(r'..\Dataset\custom_2017_2020.csv', delimiter=',', skiprows=1)
cols = ["exp_imp", "Year", "month", "ym", "Country", "Custom", "hs2", "hs4", "hs6", "hs9", "Q1", "Q2", "Value"]

df = pd.DataFrame(data, columns=cols)
features = df.iloc[:, :-1]
target = df.iloc[:, -1]

# Split the data into training and validation sets
train_size = int(0.8 * len(features))
X_train, X_val = features[:train_size], features[train_size:]
y_train, y_val = target[:train_size], target[train_size:]

# Create an instance of LinearRegression class
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Normalize the validation data using the mean and standard deviation of the training data
X_val_normalized = (X_val - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Make predictions on the validation data
val_pred = model.predict(X_val_normalized)

# Calculate validation loss
val_loss = np.mean(np.square(np.squeeze(y_val) - np.squeeze(val_pred)))
print("Validation Loss:", val_loss)

# Calculate validation accuracy
val_accuracy = 1 - np.mean(np.abs(np.squeeze(val_pred) - np.squeeze(y_val)) / np.squeeze(y_val))
print("Validation Accuracy:", abs(val_accuracy))
