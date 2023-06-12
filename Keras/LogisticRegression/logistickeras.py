import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import RandomOverSampler
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)
print(df)
df.head()


from tensorflow import keras
from tensorflow.keras import layers

class LogisticRegression:
    def __init__(self, input_dim):
        self.model = keras.Sequential()
        opti = SGD(learning_rate= 0.01)
        self.model.add(layers.Dense(1, activation='sigmoid', input_shape=(input_dim,)))
        self.model.compile(optimizer= opti, loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train, num_epochs, display_step):
      for epoch in range(num_epochs):
            # Train the model for one epoch
        history = self.model.fit(X_train, y_train, epochs=1, verbose=0)

            # Display the loss every display_staep steps
        if (epoch + 1) % display_step == 0:
          loss = history.history['loss'][0]
          print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X_test):
      return self.model.predict(X_test)
    def evaluate(self, X_test, y_test):
      _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
      return accuracy



train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 

train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 


X_train = train.drop("class", axis=1)
Y_train = train["class"]

X_test = test.drop("class", axis=1)
Y_test = test["class"]

logreg= LogisticRegression(X_train.shape[1])
logreg.fit(X_train , Y_train, 1000, 100)
X = train.drop("class", axis=1)
y = train["class"]

logreg.predict(X_test.values)

accuracy = logreg.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.4f}")
