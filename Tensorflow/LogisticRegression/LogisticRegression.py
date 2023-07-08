import numpy as np
import pandas as pd
import tensorflow as tf
import time

class LogisticRegression:
    def __init__(self, learning_rate, num_epochs, display_step, batch_size):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.display_step = display_step
        self.batch_size = batch_size
        self.num_features = None
        self.num_classes = None
        self.W = None
        self.b = None
        self.optimizer = None

    def preprocess_data(self, df):
        np.random.seed(42)
        df = df.sample(frac=1).reset_index(drop=True)
        features = df.drop('class', axis=1).values
        target = df["class"].values

        train_size = int(0.8 * len(features))
        X_train, X_val = features[:train_size], features[train_size:]
        y_train, y_val = target[:train_size], target[train_size:]

        self.num_features = X_train.shape[1]
        self.num_classes = 2

        # Reshape the target labels
        y_train = np.eye(self.num_classes)[y_train.reshape(-1)]
        y_val = np.eye(self.num_classes)[y_val.reshape(-1)]

        return X_train, y_train, X_val, y_val

    def create_variables(self):
        self.W = tf.Variable(tf.zeros([self.num_features, self.num_classes], dtype=tf.float64))
        self.b = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float64))
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)

    @tf.function
    def logistic_regression(self, inputs):
        logits = tf.matmul(tf.cast(inputs, dtype=tf.float64), self.W) + self.b
        return tf.nn.softmax(logits)

    @tf.function
    def loss_fn(self, inputs, labels):
        logits = self.logistic_regression(inputs)
        loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss_value

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            loss_value = self.loss_fn(inputs, labels)
        gradients = tape.gradient(loss_value, [self.W, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.W, self.b]))
        return loss_value

    def train(self, X_train, y_train):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            num_batches = len(X_train) // self.batch_size

            for batch in range(num_batches):
                batch_indices = np.random.choice(len(X_train), size=self.batch_size, replace=False)
                batch_features = X_train[batch_indices]
                batch_target = y_train[batch_indices]

                loss_value = self.train_step(batch_features, batch_target)

            if (epoch + 1) % self.display_step == 0:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss_value:.5f}')

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training completed in {execution_time:.2f} seconds.")

    def predict(self, X_val, y_val):
        val_pred = self.logistic_regression(X_val).numpy()
        val_pred_labels = np.argmax(val_pred, axis=1)
        val_true_labels = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_pred_labels == val_true_labels)
        print(f'Validation Accuracy: {val_accuracy:.5f}')

        # return val_pred_labels, val_true_labels

# Specify the device type as "cuda"
device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")

# Load the data
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

# Instantiate the LogisticRegression class
lr = LogisticRegression(learning_rate=0.01, num_epochs=100, display_step=10, batch_size=32)

# Preprocess the data
X_train, y_train, X_val, y_val = lr.preprocess_data(df)

# Create the variables
lr.create_variables()

# Train the model
lr.train(X_train, y_train)

# Predict on the validation set
# val_pred_labels, val_true_labels = lr.predict(X_val, y_val)

lr.predict(X_val, y_val)
