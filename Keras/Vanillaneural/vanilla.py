import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Resize images to 32x32 to match LeNet's input shape
from skimage.transform import resize

train_images_resized = np.array([resize(image, (32, 32)) for image in train_images])
test_images_resized = np.array([resize(image, (32, 32)) for image in test_images])

# Normalize pixel values to the range [0, 1]
train_images_resized = train_images_resized.astype('float32') / 255.0
test_images_resized = test_images_resized.astype('float32') / 255.0

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Create the LeNet model
def lenet_model(input_shape=(32, 32, 1), num_classes=10):
    model = models.Sequential()
    
    # Layer 1
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    
    # Output layer with softmax activation for num_classes
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Create the model
model = lenet_model(input_shape=(32, 32, 1), num_classes=10)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Reshape the data to match the model's input shape
train_images_reshaped = np.expand_dims(train_images_resized, axis=-1)
test_images_reshaped = np.expand_dims(test_images_resized, axis=-1)

# Train the model and measure the training time
start_time = time.time()
model.fit(train_images_reshaped, train_labels, batch_size=128, epochs=10, verbose=1)
training_time = time.time() - start_time

# Evaluate the model and measure the prediction time
start_time = time.time()
test_loss, test_accuracy = model.evaluate(test_images_reshaped, test_labels, verbose=0)
prediction_time = (time.time() - start_time) / len(test_images_reshaped)  # Time per image prediction

print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time per Image: {prediction_time:.5f} seconds")
print(f"Test Accuracy: {test_accuracy:.4f}")
