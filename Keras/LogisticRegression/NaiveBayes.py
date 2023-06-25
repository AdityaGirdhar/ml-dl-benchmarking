import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.model = None

    def fit(self, X_train, y_train):
        st = time.time()
        self.classes = np.unique(y_train)
        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        self.model = Sequential()
        self.model.add(Dense(num_classes, input_shape=(num_features,), activation='softmax'))
        opt = SGD(learning_rate=0.01)  
        self.model.compile(optimizer= opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10, verbose=0)
        end_time = time.time()
        print(end_time - st)
        
    def predict(self, X_test):
        start_time = time.time()
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        end_time = time.time()
        prediction_time = end_time - start_time
        return y_pred, prediction_time

# Set GPU memory growth if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the dataset and split into training and testing sets
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

X = df.drop("class", axis=1).values
y = df["class"].values

# Manual train-test split
np.random.seed(42)
shuffle_indices = np.random.permutation(len(X))
train_indices = shuffle_indices[:int(0.8 * len(X))]
test_indices = shuffle_indices[int(0.8 * len(X)):]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Create and train the Naive Bayes classifier
naive_bayes = NaiveBayes()
naive_bayes.fit(X_train, y_train)

tup = naive_bayes.predict(X_test)

# Calculate accuracy
accuracy = np.mean(tup[0] == y_test)

# print("Test Accuracy: {:.4f}".format(accuracy))
# print("Time taken in predicting: {:.4f} seconds".format(tup[1]))
