import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import cProfile
import pstats
from memory_profiler import profile

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.model = None
    @profile
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        self.model = Sequential()
        self.model.add(Dense(num_classes, input_shape=(num_features,), activation='softmax'))
        opt = SGD(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        profiler = cProfile.Profile()
        profiler.enable()

        self.model.fit(X_train, y_train, epochs=10, verbose=0)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('fit')
    @profile
    def predict(self, X_test):
        profiler = cProfile.Profile()
        profiler.enable()

        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats('predict')

        return y_pred


def run_code():
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
 
    y_pred = naive_bayes.predict(X_test)
    
    _, train_accuracy = naive_bayes.model.evaluate(X_train, y_train, verbose=0)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print("Test Accuracy: {:.4f}".format(accuracy))


run_code()
