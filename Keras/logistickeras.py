import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)
print(df)
df.head()
def scale_dataset(dataframe, oversample=False):
   X = dataframe[dataframe.columns[:-1]].values
   y = dataframe[dataframe.columns[-1]].values
   scaler = StandardScaler()
   X = scaler.fit_transform(X)

   if oversample:
     ros = RandomOverSampler()
     X, y = ros.fit_resample(X, y)

   data = np.hstack((X, np.reshape(y, (-1, 1))))
   columns = list(dataframe.columns[:-1]) + [dataframe.columns[-1]]
   scaled_df = pd.DataFrame(data, columns=columns)
   return scaled_df, X, y

train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 

X = train.drop("class", axis=1)
y = train["class"]
train, X_train, y_train = scale_dataset(train)
test, X_test, y_test = scale_dataset(test)
print(len(train[train["class"] == 1]))
print(len(train[train["class"] == 0]))
train =  df.sample(frac=0.8) 
test= df.sample(frac=0.2) 
X = train.drop("class", axis=1)
Y = train["class"]

model = Sequential()
model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x= X, y=Y , epochs=1000, verbose=1)

print(y_test[:10])
predictions= model.predict(X_test)
print(predictions[:10])