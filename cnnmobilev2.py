import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define your root directory where image folders are located.
root_directory = "/content/drive/MyDrive/GDToT/dataset 9/data0330"

# Define the list of class labels based on subdirectories.
class_labels = sorted(os.listdir(root_directory))

# Create empty lists to store image data and labels.
images = []
labels = []

# Load and preprocess a smaller subset of images from folders for quicker processing.
subset_size = 300  # You can adjust this subset size
for i, label in enumerate(class_labels):
    label_directory = os.path.join(root_directory, label)
    count = 0
    for image_filename in os.listdir(label_directory):
        if count >= subset_size:
            break
        image_path = os.path.join(label_directory, image_filename)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess for MobileNetV2
        images.append(image)
        labels.append(i)  # Use class index as label
        count += 1

# Convert image data and labels to NumPy arrays.
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets for both CNN and MobileNetV2.
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train_mobilenet, X_test_mobilenet, y_train_mobilenet, y_test_mobilenet = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define a simpler CNN model with fewer epochs.
cnn_model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_labels), activation='softmax')
])

# Compile the CNN model.
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with fewer epochs.
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=5, batch_size=32)

# Evaluate the CNN model on the test set.
test_loss_cnn, test_accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"Test Accuracy (CNN): {test_accuracy_cnn * 120:.2f}%")

# Load MobileNetV2 pre-trained on ImageNet without the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification head
mobilenet_model = keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(len(class_labels), activation='softmax')
])

# Compile the MobileNetV2 model.
mobilenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the MobileNetV2 model.
mobilenet_model.fit(X_train_mobilenet, y_train_mobilenet, epochs=5, batch_size=32)

# Evaluate the MobileNetV2 model on the test set.
test_loss_mobilenet, test_accuracy_mobilenet = mobilenet_model.evaluate(X_test_mobilenet, y_test_mobilenet)
print(f"Test Accuracy (MobileNetV2): {test_accuracy_mobilenet * 100:.2f}%")
