import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers

cols = ["frequency", "angleofattack", "chordlength", "freestreamvelocity", "suctionsidedisplacement", "soundpressure"]
data = np.loadtxt("airfoil_self_noise.dat")
#visuaize data
df = pd.DataFrame(data, columns=cols)
for label in df.columns[:-1]:
    plt.scatter(df[label], df["soundpressure"])
    plt.title(label)
    plt.ylabel("Pressure of sound")
    plt.xlabel(label)
    plt.show()

# Separate the input features (X) and target variable (y)
X = df.drop("soundpressure", axis=1)
y = df["soundpressure"]

# Split the data into training and testing sets
train = df.sample(frac=0.8, random_state=1)  # Use a fixed random_state for reproducibility
test = df.drop(train.index)

# Normalize the input features
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# Build the linear regression model using Keras
model = keras.Sequential()
model.add(layers.Dense(1, input_shape=(X.shape[1],)))

# Define the SGD optimizer with learning rate
optimizer = keras.optimizers.SGD(lr=0.01)

# Compile the model with the optimizer and loss function
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Train the model
model.fit(X, y, epochs=1000, batch_size=32)

# Evaluate the model on the test set
X_test = (test.drop("soundpressure", axis=1) - X_mean) / X_std
y_test = test["soundpressure"]
y_pred = model.predict(X_test).reshape(-1)  # Reshape to 1-dimensional array

mse = keras.metrics.mean_squared_error(y_test, y_pred)
mae = keras.metrics.mean_absolute_error(y_test, y_pred)

# Start a TensorFlow session
with tf.Session() as sess:
    mse_value = sess.run(mse)
    mae_value = sess.run(mae)

print("MSE:", mse_value)
print("MAE:", mae_value)


# plt.scatter(X_test,y_test, color= 'black')
plt.scatter(y_test,model.predict(X_test),color= "blue",linewidth=3)
plt.show()