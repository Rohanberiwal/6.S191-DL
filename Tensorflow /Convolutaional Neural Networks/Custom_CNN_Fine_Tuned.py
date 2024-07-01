import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}

# Function to load annotations and images into dictionaries
def load_annotations_into_dict(annotation_file, root_dir, target_dict):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    for filename, attributes in data.items():
        img_name = attributes['filename']  
        img_path = os.path.join(root_dir, img_name)
        boxes = []
        for region in attributes['regions']:
            shape_attr = region['shape_attributes']
            x = shape_attr['x']
            y = shape_attr['y']
            width = shape_attr['width']
            height = shape_attr['height']
            boxes.append([x, y, x + width, y + height])
        target_dict[img_path] = boxes

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image)  # Convert PIL image to numpy array
    image = tf.keras.applications.resnet50.preprocess_input(image)  # Preprocess image for ResNet50
    return image

# Load images and labels from dictionaries
def load_images_and_labels(image_dict):
    X = []
    y = []
    for img_path, boxes in image_dict.items():
        img = preprocess_image(img_path)
        X.append(img)
        if 'mitotic' in img_path.lower():
            y.append(1)
        else:
            y.append(0)
    return np.array(X), np.array(y)

# Define root directory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Load annotations into dictionaries
load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

# Load images and labels from dictionaries
X_mitotic, y_mitotic = load_images_and_labels(standard_dict_mitotic)
X_non_mitotic, y_non_mitotic = load_images_and_labels(standard_dict_non_mitotic)
X = np.concatenate([X_mitotic, X_non_mitotic], axis=0)
y = np.concatenate([y_mitotic, y_non_mitotic], axis=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_custom_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

custom_model = create_custom_cnn((224, 224, 3))
custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = custom_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)
loss, accuracy = custom_model.evaluate(X_val, y_val)
print(f"Validation accuracy: {accuracy * 100:.2f}%")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
