import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import json
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from torchvision.models import resnet50
from torchvision.ops import roi_pool
import torch.nn.functional as F
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import os
import json
import torch
import torchvision
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import tensorflow as tf
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}
import os
from torchvision.ops import roi_pool
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F 
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.patches as patches

# Constants
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\MitoticFolder"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\NonMitoticFolder"
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}

# Load mitotic and non-mitotic annotations
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
                boxes.append([x, y, width, height])
            target_dict[img_path] = boxes

    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")

load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

# Convert keys in dictionaries to match image file extensions
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

# Helper function to plot patches with bounding boxes
def plot_patches(image_path, boxes):
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        for xmin, ymin, width, height in boxes:
            patch = image.crop((xmin, ymin, xmin + width, ymin + height))
            patch = patch.resize((100, 100))
            ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error plotting patches for image {image_path}: {e}")

# Plot mitotic and non-mitotic patches together
def plot_boxes_together(dict_mitotic, dict_non_mitotic):
    common_keys = set(dict_mitotic.keys()).intersection(dict_non_mitotic.keys())

    for key in common_keys:
        try:
            image = Image.open(key)
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            ax = plt.gca()

            for xmin, ymin, width, height in dict_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            for xmin, ymin, width, height in dict_non_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)

            plt.title(f"Image: {os.path.basename(key)}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error processing image {key}: {e}")

# Save patches from images
def save_patches(image_dict, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    patch_count = 0

    for image_path, boxes in image_dict.items():
        try:
            image = Image.open(image_path)

            for xmin, ymin, width, height in boxes:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((224, 224))

                patch_save_path = os.path.join(save_dir, f"{prefix}_patch_{patch_count}.jpg")
                patch.save(patch_save_path)

                patch_count += 1

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

save_patches(standard_dict_mitotic, mitotic_save_dir, "mitotic")
save_patches(standard_dict_non_mitotic, non_mitotic_save_dir, "non_mitotic")
print("Patches saved successfully.")

# Load file paths
def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

list_mitotic = get_file_paths(mitotic_save_dir)
list_non_mitotic = get_file_paths(non_mitotic_save_dir)
print("List of mitotic patches:", list_mitotic)
print("List of non-mitotic patches:", list_non_mitotic)

# Split data into train and validation sets
train_ratio = 0.7
validation_ratio = 0.3

num_mitotic = len(list_mitotic)
num_non_mitotic = len(list_non_mitotic)

num_mitotic_train = int(train_ratio * num_mitotic)
num_mitotic_val = num_mitotic - num_mitotic_train

num_non_mitotic_train = int(train_ratio * num_non_mitotic)
num_non_mitotic_val = num_non_mitotic - num_non_mitotic_train

mitotic_train_paths = list_mitotic[:num_mitotic_train]
mitotic_val_paths = list_mitotic[num_mitotic_train:]

non_mitotic_train_paths = list_non_mitotic[:num_non_mitotic_train]
non_mitotic_val_paths = list_non_mitotic[num_non_mitotic_train:]

print("Number of mitotic patches (train):", num_mitotic_train)
print("Number of mitotic patches (val):", num_mitotic_val)
print("Number of non-mitotic patches (train):", num_non_mitotic_train)
print("Number of non-mitotic patches (val):", num_non_mitotic_val)

# Helper function to load image data and labels
def load_image_and_label(image_path, label, target_size=(224, 224)):
    try:
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        return image_array, label

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

# Prepare training data
train_data = []
train_labels = []

for image_path in mitotic_train_paths:
    image_array, label = load_image_and_label(image_path, 1)
    if image_array is not None:
        train_data.append(image_array)
        train_labels.append(label)

for image_path in non_mitotic_train_paths:
    image_array, label = load_image_and_label(image_path, 0)
    if image_array is not None:
        train_data.append(image_array)
        train_labels.append(label)

# Prepare validation data
val_data = []
val_labels = []

for image_path in mitotic_val_paths:
    image_array, label = load_image_and_label(image_path, 1)
    if image_array is not None:
        val_data.append(image_array)
        val_labels.append(label)

for image_path in non_mitotic_val_paths:
    image_array, label = load_image_and_label(image_path, 0)
    if image_array is not None:
        val_data.append(image_array)
        val_labels.append(label)

# Convert to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

print("Training data shape:", train_data.shape)
print("Training labels shape:", train_labels.shape)
print("Validation data shape:", val_data.shape)
print("Validation labels shape:", val_labels.shape)

# Define BB regx model
bb_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='linear')  # Output 4 for (xmin, ymin, width, height)
])

bb_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Fit BB regx model
bb_model.fit(train_data, standard_dict_mitotic, epochs=50, validation_data=(val_data, standard_dict_mitotic))

# Evaluate BB regx model
bb_loss, bb_mae = bb_model.evaluate(val_data, standard_dict_mitotic)
print(f"BB regx model evaluation - Loss: {bb_loss}, MAE: {bb_mae}")

# Define classification model
classification_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit classification model
classification_history = classification_model.fit(train_data, train_labels, epochs=200, validation_data=(val_data, val_labels))

# Plot classification model training history
plt.plot(classification_history.history['accuracy'], label='train_accuracy')
plt.plot(classification_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("This is the complete code.")
