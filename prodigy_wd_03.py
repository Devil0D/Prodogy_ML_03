import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Path to your own dataset
train_dir = './dataset/train'  # Directory for training images
cat_dir = os.path.join(train_dir, 'cat')  # Path for the cat images
dog_dir = os.path.join(train_dir, 'dog')  # Path for the dog images

# Normalize paths for proper reading
cat_dir = os.path.normpath(cat_dir)
dog_dir = os.path.normpath(dog_dir)

# Debugging: print the absolute paths
print(f"Absolute path for cat directory: {os.path.abspath(cat_dir)}")
print(f"Absolute path for dog directory: {os.path.abspath(dog_dir)}")

# Function to load images and their labels
def load_images_and_labels(cat_dir, dog_dir, max_images=1500):
    images = []
    labels = []
    
    # Load cat images (label: 0)
    cat_images = os.listdir(cat_dir)[:max_images]  # Limit the number of images to 1500
    for img_name in cat_images:
        img_path = os.path.join(cat_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            images.append(img.flatten())  # Flatten the image to a 1D array
            labels.append(0)  # Label for cats is 0

    # Load dog images (label: 1)
    dog_images = os.listdir(dog_dir)[:max_images]  # Limit the number of images to 1500
    for img_name in dog_images:
        img_path = os.path.join(dog_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            images.append(img.flatten())  # Flatten the image to a 1D array
            labels.append(1)  # Label for dogs is 1

    return np.array(images), np.array(labels)

# Load the training data and labels
images, labels = load_images_and_labels(cat_dir, dog_dir, max_images=1500)

# If no images were loaded, exit the program
if len(images) == 0 or len(labels) == 0:
    print("No images were loaded. Exiting...")
    exit()

# Scale the images to improve SVM performance
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)  # Scale the features (flattened images)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_scaled, labels, test_size=0.2, random_state=42)

# Train the SVM model with RBF kernel
svm_model = svm.SVC(kernel='rbf', C=1, gamma='scale')  # RBF kernel is often better for images
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

# Cross-validation score (for better understanding of model performance)
cv_scores = cross_val_score(svm_model, images_scaled, labels, cv=5)
print(f'Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%')

# Visualize some test predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for ax, idx in zip(axes, range(5)):
    ax.imshow(X_test[idx].reshape(64, 64), cmap='gray')  # Reshape back to 64x64 for visualization
    ax.set_title(f"Predicted: {'cat' if y_pred[idx] == 0 else 'dog'}")
    ax.axis('off')
plt.show()
