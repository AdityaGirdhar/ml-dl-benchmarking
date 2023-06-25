import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import time

class LogisticRegression:
    def __init__(self, input_dim):
        self.model = keras.Sequential()
        opti = SGD(learning_rate=0.01)  # Set the learning rate using 'lr' parameter
        self.model.add(layers.Dense(1, activation='sigmoid', input_shape=(input_dim,)))
        self.model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
        

    def fit(self, X_train, y_train, num_epochs, display_step):
      # start_time= time.time()
      for epoch in range(num_epochs):
            # Train the model for one epoch
        history = self.model.fit(X_train, y_train, epochs=1, verbose=0)

            # Display the loss every display_staep steps
        if (epoch + 1) % display_step == 0:
          loss = history.history['loss'][0]
          print("Epoch " + str(epoch+1) + ", Loss: " + str(loss))

          
      

    def predict(self, X_test):
      start_time= time.time()
      val = self.model.predict(X_test)
      end_time= time.time()
      train_time= end_time - start_time
      return val , train_time
    def evaluate(self, X_test, y_test):
      _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
      return accuracy

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 

train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 


X_train = train.drop("class", axis=1)
Y_train = train["class"]

X_test = test.drop("class", axis=1)
Y_test = test["class"]

logreg= LogisticRegression(X_train.shape[1])
pt=logreg.fit(X_train , Y_train, 100, 10)
# print("training time" , pt)
X = train.drop("class", axis=1)
y = train["class"]

tup = logreg.predict(X_test)

accuracy = logreg.evaluate(X_test, Y_test)
print("Test Accuracy: " + str(accuracy))
print("predicting time " , tup[1])