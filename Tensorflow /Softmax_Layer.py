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

# Split paths into training and validation sets
train_ratio = 0.7
validation_ratio = 0.3

num_mitotic = len(list_mitotic)
num_non_mitotic = len(list_non_mitotic)

num_mitotic_train = int(train_ratio * num_mitotic)
num_mitotic_val = num_mitotic - num_mitotic_train

num_non_mitotic_train = int(train_ratio * num_non_mitotic)
num_non_mitotic_val = num_non_mitotic - num_non_mitotic_train

# Assign training and validation sets for mitotic and non-mitotic patches
mitotic_train_paths = list_mitotic[:num_mitotic_train]
mitotic_val_paths = list_mitotic[num_mitotic_train:]

non_mitotic_train_paths = list_non_mitotic[:num_non_mitotic_train]
non_mitotic_val_paths = list_non_mitotic[num_non_mitotic_train:]

print("Number of mitotic patches (train):", num_mitotic_train)
print("Number of mitotic patches (val):", num_mitotic_val)
print("Number of non-mitotic patches (train):", num_non_mitotic_train)
print("Number of non-mitotic patches (val):", num_non_mitotic_val)

# Function to load image and label
def load_image_and_label(image_path, label, target_size=(224, 224)):
    try:
        # Load image from path and resize
        image = load_img(image_path, target_size=target_size)

        # Convert image to array and normalize pixel values
        image_array = img_to_array(image)
        image_array = image_array / 255.0

        return image_array, label

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

import numpy as np
import tensorflow as tf
from PIL import Image

# Define manual augmentation function
def manual_augmentation(image_array):
    # Perform manual augmentation (e.g., flipping horizontally)
    flipped_image = tf.image.flip_left_right(image_array)
    return flipped_image

def generate_data_batches(image_paths, label, batch_size=32):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_labels = []

            for image_path in batch_paths:
                image_array, _ = load_image_and_label(image_path, label)
                if image_array is not None:
                    # Apply manual augmentation
                    augmented_image = manual_augmentation(image_array)
                    batch_images.append(augmented_image)
                    batch_labels.append(label)

            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            yield batch_images, batch_labels

# Define batch size and number of epochs
batch_size = 32
epochs = 500

# Update labels to categorical format
mitotic_train_labels = [1] * len(mitotic_train_paths)
mitotic_val_labels = [1] * len(mitotic_val_paths)
non_mitotic_train_labels = [0] * len(non_mitotic_train_paths)
non_mitotic_val_labels = [0] * len(non_mitotic_val_paths)

# Combine mitotic and non-mitotic paths and labels
train_paths = mitotic_train_paths + non_mitotic_train_paths
train_labels = mitotic_train_labels + non_mitotic_train_labels
val_paths = mitotic_val_paths + non_mitotic_val_paths
val_labels = mitotic_val_labels + non_mitotic_val_labels

# Shuffle the training and validation sets
train_data = list(zip(train_paths, train_labels))
val_data = list(zip(val_paths, val_labels))
np.random.shuffle(train_data)
np.random.shuffle(val_data)
train_paths, train_labels = zip(*train_data)
val_paths, val_labels = zip(*val_data)

# Update data generator to handle both classes
def generate_data_batches(image_paths, labels, batch_size=32):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_images = []

            for image_path in batch_paths:
                image_array, _ = load_image_and_label(image_path, None)
                if image_array is not None:
                    # Apply manual augmentation
                    augmented_image = manual_augmentation(image_array)
                    batch_images.append(augmented_image)

            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            # Convert labels to categorical
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

            yield batch_images, batch_labels

# Build the CNN model with softmax activation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model with categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Count the number of training and validation steps per epoch
train_steps_per_epoch = len(train_paths) // batch_size
val_steps_per_epoch = len(val_paths) // batch_size

# Create data generators for training and validation
train_generator = generate_data_batches(train_paths, train_labels, batch_size=batch_size)
val_generator = generate_data_batches(val_paths, val_labels, batch_size=batch_size)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator, steps=val_steps_per_epoch)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Function to plot feature maps for an image patch
def plot_feature_maps(model, image_path):
    try:
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        # Get feature maps for each layer in the model
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(image_array)

        # Plot feature maps
        layer_names = [layer.name for layer in model.layers]
        for layer_name, layer_activation in zip(layer_names, activations):
            if 'conv2d' in layer_name.lower():
                print(f"Feature maps for layer: {layer_name}")
                num_filters = layer_activation.shape[-1]
                display_grid = np.zeros((8, 8 * num_filters))

                for filter_index in range(num_filters):
                    channel_image = layer_activation[0, :, :, filter_index]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[:, filter_index * 8: (filter_index + 1) * 8] = channel_image

                scale = 1.0 / 4.0
                plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')

                plt.show()

    except Exception as e:
        print(f"Error plotting feature maps for image {image_path}: {e}")

# Example: Plot feature maps for the first non-mitotic patch
if len(non_mitotic_train_paths) > 0:
    example_image_path = non_mitotic_train_paths[0]
    plot_feature_maps(model, example_image_path)
else:
    print("No non-mitotic patches available for feature map visualization.")
