import numpy as np
import pandas as pd
import tensorflow as tf
import time

class NaiveBayesClassifier:
    def __init__(self):
        self.tf_seed = 42
        self.np_seed = 42

    def train(self, X_train, y_train):
        np.random.seed(self.np_seed)
        tf.random.set_seed(self.tf_seed)
        
        class_counts = np.bincount(y_train.astype(int))
        self.class_priors = class_counts / len(y_train)

        self.num_features = X_train.shape[1]
        self.feature_probs = np.zeros((self.num_features, len(self.class_priors)))

        for feature_idx in range(self.num_features):
            for class_idx in range(len(self.class_priors)):
                feature_values = X_train[y_train == class_idx, feature_idx]
                self.feature_probs[feature_idx, class_idx] = np.mean(feature_values)

    def predict(self, X_val):
        epsilon = 1e-10
        val_predictions = []

        for sample in X_val:
            class_scores = []

            for class_idx in range(len(self.class_priors)):
                class_score = np.log(self.class_priors[class_idx])

                for feature_idx in range(self.num_features):
                    feature_value = sample[feature_idx]
                    class_score += np.log(self.feature_probs[feature_idx, class_idx] + epsilon) if feature_value == 1 else np.log(1 - self.feature_probs[feature_idx, class_idx] + epsilon)

                class_scores.append(class_score)

            predicted_class = np.argmax(class_scores)
            val_predictions.append(predicted_class)
        
        return val_predictions
    
# Specify the device type as "cuda"
device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

tf.random.set_seed(42)
np.random.seed(42)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Normalize the feature values
df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()

# Split the data into features and target
features = df.drop('class', axis=1).values
target = df["class"].values

# Split the data into training and validation sets
train_size = int(0.8 * len(features))
X_train, X_val = features[:train_size], features[train_size:]
y_train, y_val = target[:train_size], target[train_size:]

start_time = time.time()

classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)
val_predictions = classifier.predict(X_val)

# Calculate accuracy on the validation set
accuracy = np.mean(val_predictions == y_val.astype(int))

end_time = time.time()
execution_time = end_time - start_time

print("Accuracy:", accuracy)
print("Execution time: {:.2f} seconds".format(execution_time))
