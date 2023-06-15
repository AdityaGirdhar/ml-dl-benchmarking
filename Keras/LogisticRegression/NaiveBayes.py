import numpy as np
import pandas as pd
import time

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.means = None
        self.variances = None
        self.classes = None

    def fit(self, X_train, y_train):
        # start_time = time.time()
        self.classes = np.unique(y_train)
        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        self.prior = np.zeros(num_classes)
        self.means = np.zeros((num_classes, num_features))
        self.variances = np.zeros((num_classes, num_features))

        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.prior[i] = X_c.shape[0] / X_train.shape[0]
            self.means[i] = np.mean(X_c, axis=0)
            self.variances[i] = np.var(X_c, axis=0)

    


    def predict(self, X_test):
        start_time = time.time()
        y_pred = np.zeros(X_test.shape[0])

        for i, x in enumerate(X_test):
            posteriors = []
            for j, c in enumerate(self.classes):
                prior = np.log(self.prior[j])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[j], self.variances[j])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred[i] = self.classes[np.argmax(posteriors)]
        end_time = time.time()  # Stop the timer
        prediction_time = end_time - start_time
        return y_pred , prediction_time

    def gaussian_pdf(self, x, mean, variance):
        exponent = np.exp(-(x - mean) ** 2 / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent


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
#print("Time taken to train data", timein)
# Predict on the test set
tup = naive_bayes.predict(X_test)

# Calculate accuracy
accuracy = np.mean(tup[0] == y_test)
print("Test Accuracy: {:.4f}".format(accuracy))
print("time taken in predicting is " , tup[1])