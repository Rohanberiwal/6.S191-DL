import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define paths and directories
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\MitoticFolder"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\NonMitoticFolder"

# Load and print annotations into dictionaries
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}

def print_mitotic(json_mitotic):
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  # Store boundary boxes directly

def print_filename_bbox(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  # Store boundary boxes directly

print_mitotic(mitotic_annotation_file)
print_filename_bbox(non_mitotic_annotation_file)

# Function to modify dictionary keys inplace
def modify_dict_inplace(standard_dict, root):
    keys_to_remove = []
    keys_to_add = []

    for key in standard_dict.keys():
        key_new = key.replace('.jpeg', '.jpg')
        image_key = os.path.join(root, key_new)

        keys_to_remove.append(key)
        keys_to_add.append(image_key)

    for old_key, new_key in zip(keys_to_remove, keys_to_add):
        standard_dict[new_key] = standard_dict.pop(old_key)

modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)

# Function to load annotations into dictionary
def load_annotations_into_dict(annotation_file, root_dir, target_dict):
    try:
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
                boxes.append([x, y, width, height])  # Append the coordinates directly
            target_dict[img_path] = boxes

    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")

load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

import matplotlib.patches as patches

def plot_patches(image_path, boxes):
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        for xmin, ymin, width, height in boxes:
            patch = image.crop((xmin, ymin, xmin + width, ymin + height))
            patch = patch.resize((100, 100))  # Resize for display
            ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error plotting patches for image {image_path}: {e}")

# Function to plot mitotic and non-mitotic boxes together on an image
def plot_boxes_together(dict_mitotic, dict_non_mitotic):
    # Iterate over keys that are common in both dictionaries
    common_keys = set(dict_mitotic.keys()).intersection(dict_non_mitotic.keys())

    for key in common_keys:
        try:
            # Load the image
            image = Image.open(key)
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            ax = plt.gca()

            # Plot mitotic bounding boxes as small patches
            for xmin, ymin, width, height in dict_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))  # Resize for display
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # Plot non-mitotic bounding boxes as small patches
            for xmin, ymin, width, height in dict_non_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))  # Resize for display
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)

            plt.title(f"Image: {os.path.basename(key)}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error processing image {key}: {e}")

# Function to save patches to directories
def save_patches(image_dict, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    patch_count = 0

    for image_path, boxes in image_dict.items():
        try:
            # Load the image
            image = Image.open(image_path)

            for xmin, ymin, width, height in boxes:
                # Extract patch based on bounding box coordinates
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))

                # Resize patch to match model input size (e.g., 224x224)
                patch = patch.resize((224, 224))

                # Save the patch
                patch_save_path = os.path.join(save_dir, f"{prefix}_patch_{patch_count}.jpg")
                patch.save(patch_save_path)

                patch_count += 1

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Get file paths of saved patches
def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# Save patches for mitotic and non-mitotic images
# save_patches(standard_dict_mitotic, mitotic_save_dir, "mitotic")
# save_patches(standard_dict_non_mitotic, non_mitotic_save_dir, "non_mitotic")
print("Patches saved successfully.")

# Get file paths of saved patches
list_mitotic = get_file_paths(mitotic_save_dir)
list_non_mitotic = get_file_paths(non_mitotic_save_dir)
print("List of mitotic patches:", list_mitotic)
print("List of non-mitotic patches:", list_non_mitotic)

# Label lists for mitotic and non-mitotic patches
labels_mitotic = [1] * len(list_mitotic)
labels_non_mitotic = [0] * len(list_non_mitotic)

# Combine lists and labels
X = np.array(list_mitotic + list_non_mitotic)
y = np.array(labels_mitotic + labels_non_mitotic)

# Shuffle combined data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split data into training and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Data generator function
def data_generator(file_paths, labels, batch_size=32, augment=False):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = []

            for path in batch_paths:
                img = load_img(path, target_size=(224, 224))
                img = img_to_array(img)
                img /= 255.0
                images.append(img)

            images = np.array(images)
            batch_labels = np.array(batch_labels)

            if augment:
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
                images, batch_labels = next(datagen.flow(images, batch_labels, batch_size=batch_size))

            yield images, batch_labels

# Define model
def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
batch_size = 32
train_gen = data_generator(X_train, y_train, batch_size=batch_size, augment=True)
val_gen = data_generator(X_val, y_val, batch_size=batch_size)

history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_gen,
    validation_steps=len(X_val) // batch_size,
    epochs=100
)

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)
