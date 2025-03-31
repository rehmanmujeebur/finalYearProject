import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import time

# Define your root directory where image folders are located.
root_directory = "/content/drive/MyDrive/GDToT/dataset 1/data0330"

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

# Apply PCA with fewer components for quicker dimensionality reduction.
pca = PCA(n_components=20)  # You can adjust the number of components as needed for speed
X_train_pca = pca.fit_transform(X_train.reshape(-1, 100 * 100 * 3))
X_test_pca = pca.transform(X_test.reshape(-1, 100 * 100 * 3))

# Define a simpler CNN model.
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_labels), activation='softmax')
])

# Compile the CNN model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with fewer epochs.
start_time_cnn = time.time()
model.fit(X_train, y_train, epochs=3, batch_size=32)
end_time_cnn = time.time()

# Evaluate the CNN model on the test set quickly.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy (CNN): {test_accuracy * 120:.2f}%")
print(f"Time taken for CNN evaluation: {end_time_cnn - start_time_cnn:.4f} seconds")

# Define the KNN model with fewer neighbors.
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors (K) as needed

# Fit the KNN model with training data after PCA.
start_time_knn = time.time()
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)
end_time_knn = time.time()

# Calculate accuracy for KNN.
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Test Accuracy (KNN): {accuracy_knn * 100:.2f}%")
print(f"Time taken for KNN evaluation: {end_time_knn - start_time_knn:.4f} seconds")
