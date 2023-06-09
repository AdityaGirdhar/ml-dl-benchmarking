import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# Load the data and create the DataFrame
cols = ["frequency", "angleofattack", "chordlength", "freestreamvelocity", "suctionsidedisplacement", "soundpressure"]
data = np.loadtxt("/content/airfoil_self_noise.dat")
df = pd.DataFrame(data, columns=cols)

# Separate the input features (X) and target variable (y)
X = df.drop("soundpressure", axis=1)
y = df["soundpressure"]

# Split the data into training and testing sets
train = df.sample(frac=0.8, random_state=1)  # Use a fixed random_state for reproducibility
test = df.drop(train.index)

# Normalize the input features
X_mean = X.mean()
X_std = X.std()
X_train = (train.drop("soundpressure", axis=1) - X_mean) / X_std
y_train = train["soundpressure"]
X_test = (test.drop("soundpressure", axis=1) - X_mean) / X_std
y_test = test["soundpressure"]

# Define the linear regression model using Keras
model = keras.Sequential()
model.add(layers.Dense(1, input_shape=(X_train.shape[1],)))

# Define the SGD optimizer with learning rate
optimizer = keras.optimizers.SGD(learning_rate=0.01)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Training loop
num_epochs = 10
display_step = 100
for epoch in range(num_epochs):
    # Forward pass
    with tf.GradientTape() as tape:
        pred = model(X_train, training=True)
        loss_value = model.loss(y_train, pred)
    
    # Calculate gradients
    gradients = tape.gradient(loss_value, model.trainable_variables)
    
    # Update weights and bias
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Print loss every display_step epochs
    if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss_value))

mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

