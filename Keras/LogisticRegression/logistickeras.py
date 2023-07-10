from memory_profiler import profile
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import cProfile
import pstats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


# profiled memory , time profiler also done , divided data as well , train and test data also splited
 
@profile
def run_code():
   

    class LogisticRegression:
        def __init__(self, input_dim):
            self.model = keras.Sequential()
            opti = SGD(learning_rate=0.01)
            self.model.add(layers.Dense(1, activation='sigmoid', input_shape=(input_dim,)))
            self.model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])

        def fit(self, X_train, y_train, num_epochs, display_step):
            self.train_accuracy = []

            profiler = cProfile.Profile()
            profiler.enable()

            for epoch in range(num_epochs):
                history = self.model.fit(X_train, y_train, epochs=1, verbose=0)
                self.train_accuracy.append(history.history['accuracy'][0])

            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            stats.sort_stats('tottime')
            stats.print_stats('fit')

        def predict(self, X_test):
            profiler = cProfile.Profile()
            profiler.enable()

            val = self.model.predict(X_test)

            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            stats.sort_stats('tottime')
            stats.print_stats('predict')

            return val

        def evaluate(self, X_test, y_test):
            _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            return accuracy

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("magic04.data", names=cols)
    df["class"] = (df["class"] == "g").astype(int)

    train = df.sample(frac=0.8)
    test = df.sample(frac=0.2)

    X_train = train.drop("class", axis=1)
    Y_train = train["class"]

    X_test = test.drop("class", axis=1)
    Y_test = test["class"]

    logreg = LogisticRegression(X_train.shape[1])
    logreg.fit(X_train, Y_train, 100, 10)

    accuracy = logreg.evaluate(X_test, Y_test)
    predictions = logreg.predict(X_test)

    print("Test Accuracy:", accuracy)
    print("Training Accuracy:", logreg.train_accuracy[-1])


run_code()
