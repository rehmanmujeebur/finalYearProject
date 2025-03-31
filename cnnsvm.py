import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(100, 100))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)
        labels.append(i)  # Use class index as label
        count += 1

# Convert image data and labels to NumPy arrays.
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define a simpler CNN model with fewer epochs.
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_labels), activation='softmax')
])

# Compile the CNN model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with fewer epochs.
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate the CNN model on the test set.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy (CNN): {test_accuracy * 120:.2f}%")

# Define the SVM model with linear kernel.
svm = SVC(kernel='linear')

# Reshape the image data for SVM.
X_train_reshaped = X_train.reshape((len(X_train), -1))
X_test_reshaped = X_test.reshape((len(X_test), -1))

# Fit the SVM model with training data.
svm.fit(X_train_reshaped, y_train)

# Predict labels for test set using SVM.
y_pred_svm = svm.predict(X_test_reshaped)

# Calculate accuracy for SVM.
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Test Accuracy (SVM): {accuracy_svm * 100:.2f}%")
