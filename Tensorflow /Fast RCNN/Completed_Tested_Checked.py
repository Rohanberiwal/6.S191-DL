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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

floattype = torch.cuda.FloatTensor

import numpy as np
import torch

import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities

class TorchROIAlign3D(object):
    
    def __init__(self, output_size, scaling_factor):
        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_align(self, features, scaled_proposal):
        _, num_channels, h, w = features.shape

        xp0, yp0, xp1, yp1 = scaled_proposal
        p_width = xp1 - xp0
        p_height = yp1 - yp0

        w_stride = p_width / self.output_size[1]
        h_stride = p_height / self.output_size[0]

        interp_features = torch.zeros((num_channels, self.output_size[0], self.output_size[1]))

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                x_bin_strt = i * w_stride + xp0
                y_bin_strt = j * h_stride + yp0

                x1 = torch.Tensor([x_bin_strt + 0.25 * w_stride])
                x2 = torch.Tensor([x_bin_strt + 0.75 * w_stride])
                y1 = torch.Tensor([y_bin_strt + 0.25 * h_stride])
                y2 = torch.Tensor([y_bin_strt + 0.75 * h_stride])

                for c in range(num_channels):
                    img = features[0, c]
                    v1 = bilinear_interpolate(img, x1, y1)
                    v2 = bilinear_interpolate(img, x1, y2)
                    v3 = bilinear_interpolate(img, x2, y1)
                    v4 = bilinear_interpolate(img, x2, y2)

                    interp_features[c, j, i] = (v1 + v2 + v3 + v4) / 4

        return interp_features

    def __call__(self, feature_layer, proposals):
        _, num_channels, _, _ = feature_layer.shape

        scaled_proposals = torch.zeros_like(proposals)
        scaled_proposals[:, 0] = proposals[:, 0] * self.scaling_factor
        scaled_proposals[:, 1] = proposals[:, 1] * self.scaling_factor
        scaled_proposals[:, 2] = proposals[:, 2] * self.scaling_factor
        scaled_proposals[:, 3] = proposals[:, 3] * self.scaling_factor

        res = torch.zeros((len(proposals), num_channels, self.output_size[0], self.output_size[1]))
        for idx in range(len(scaled_proposals)):
            proposal = scaled_proposals[idx]
            res[idx] = self._roi_align(feature_layer, proposal)

        return res

def bilinear_interpolate(img, x, y):
    x = x.clamp(min=0, max=img.shape[1] - 1)
    y = y.clamp(min=0, max=img.shape[0] - 1)
    x0 = x.floor().to(torch.int64)
    x1 = (x0 + 1).clamp(max=img.shape[1] - 1)
    y0 = y.floor().to(torch.int64)
    y1 = (y0 + 1).clamp(max=img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    interpolated_values = (Ia * wa) + (Ib * wb) + (Ic * wc) + (Id * wd)
    return interpolated_values



def selective_search(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def get_proposals(image_path):

    image = cv2.imread(image_path)
    proposals = selective_search(image)
    return proposals

def ground_truth_func(json_file):

    with open(json_file, 'r') as f:
        data = json.load(f)
    bounding_boxes = []
    for image_filename, image_data in data.items():
        regions = image_data.get('regions', [])
        for region in regions:
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', None)
            y = shape_attributes.get('y', None)
            width = shape_attributes.get('width', None)
            height = shape_attributes.get('height', None)
            if x is not None and y is not None and width is not None and height is not None:
                bounding_box = [x, y, x + width, y + height]  
                bounding_boxes.append(bounding_box)
    return bounding_boxes

def print_mitotic(json_mitotic):
    print("This is the mitotic printer function")
    universal_list = []
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, width={width}, height={height}")
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    print(universal_list)
    return standard_dict_mitotic

def print_filename_bbox(json_file):

    print("This is the printer filename function for non-mitotic")
    universal_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, width={width}, height={height}")
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    print(universal_list)
    return standard_dict_non_mitotic



def extract_bounding_boxes(annotation_file, root_dir):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    annotations = {}
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
        annotations[img_path] = boxes
    
    return annotations


transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def tensor_image(image):
    image = T.ToTensor()(image)
    return image

def load_images_and_annotations(root_dir, mitotic_dict, non_mitotic_dict):
    X = []
    y = []
    for filename, mitotic_boxes in mitotic_dict.items():
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to (224, 224)
        img = np.array(img)  # Convert PIL.Image to numpy array
        X.append(img)
        y.append(1)  # Mitotic class label
        
    for filename, non_mitotic_boxes in non_mitotic_dict.items():
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to (224, 224)
        img = np.array(img)  # Convert PIL.Image to numpy array
        X.append(img)
        y.append(0)  # Non-mitotic class label
        
    return np.array(X), np.array(y)

def extract_features(image_path):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = base_model.predict(image)
    return features

def extract_filenames_from_json(json_file, root):
    with open(json_file) as f:
        data = json.load(f)
    
    filename_list = []
    for filename, attributes in data.items():
        filename = filename.replace('.jpeg', '.jpg')
        img_name = attributes['filename'] 
        img_path = os.path.join(root, img_name)  
        file_path  = img_path
        filename_list.append(img_path)
        
    return filename_list

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

# Define root directory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Extract filenames and modify dicts
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
mitotic_output = print_mitotic(mitotic_annotation_file)
non_mitotic_output = print_filename_bbox(non_mitotic_annotation_file)

modify_dict_inplace(standard_dict_non_mitotic, root)

modify_dict_inplace(standard_dict_non_mitotic, root)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



# Define rootdirectory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Load annotations
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)

# Modify dictionaries inplace
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
print(standard_dict_non_mitotic)
print(standard_dict_mitotic)
# Load images and annotations
X, y = load_images_and_annotations(root, standard_dict_mitotic, standard_dict_non_mitotic)
print(X)
print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

batch_size = 4
steps_per_epoch = len(X_train) // batch_size
epochs = 10

test_img_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Experiment with different units
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Adjusting learning rate and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Experiment with different learning rates
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

filepath = 'best_model.keras'  
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
]
# Training the model with callbacks
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
print(history.history.keys())
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
print("CNN is tuned")
print("Starting the code")


bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******\n")

print("****** SelectiVE search ********\n")
proposals = get_proposals(image_path)
print("The proposal regions are:", proposals)
print("Number of Region Proposals:", len(proposals))
print("\n****** SelectiVE search ********\n")

print("****** CNN Layer ********\n")
features = extract_features(image_path)
print(features)
print("\n****** CNN Layer ********\n")

print("****** ROI Pooling Layer ********\n")
output_size =  (7,7)
scaling_factor =  1.0
roi_align_layer = TorchROIAlign3D(output_size, scaling_factor)
print("This is the maps",features)


    
    
    
    





## FCN pipeline code 
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import concurrent.futures
import time
def main_regression() :
    # Constants
    mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
    non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"
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

    # Load annotations into dictionaries
    root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
    mitotic_annotation_file = 'mitotic.json'
    non_mitotic_annotation_file = 'NonMitotic.json'

    load_annotations_into_dict(mitotic_annotation_file ,root, standard_dict_mitotic)
    load_annotations_into_dict(non_mitotic_annotation_file  , root, standard_dict_non_mitotic)

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
    print(standard_dict_mitotic)
    print("\n")
    print(standard_dict_non_mitotic)
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

    # Prepare training data for classification
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

    # Prepare validation data for classification
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

    # Convert to numpy arrays for classification
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)
    print("Validation data shape:", val_data.shape)
    print("Validation labels shape:", val_labels.shape)

    # Build a simple CNN model for classification
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the classification model
    history = model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_data=(val_data, val_labels))

    # Evaluate the classification model
    loss, accuracy = model.evaluate(val_data, val_labels)
    print(f"Validation loss (classification): {loss:.4f}")
    print(f"Validation accuracy (classification): {accuracy:.4f}")

    # Plot classification model training history
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy (Classification)')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss (Classification)')

    plt.tight_layout()
    plt.show()

    print("This is the train phase")
    train_bb_data = []
    train_bb_targets = []

    # Process mitotic images
    for image_path, boxes in standard_dict_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                train_bb_data.append(image_array)
                train_bb_targets.append(target)

    # Process non-mitotic images
    for image_path, boxes in standard_dict_non_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                train_bb_data.append(image_array)
                train_bb_targets.append(target)


    print("This is hte validation Phase")
    val_bb_data = []
    val_bb_targets = []

    # Process mitotic images
    for image_path, boxes in standard_dict_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                val_bb_data.append(image_array)
                val_bb_targets.append(target)

    # Process non-mitotic images
    for image_path, boxes in standard_dict_non_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                val_bb_data.append(image_array)
                val_bb_targets.append(target)

    # Convert to numpy arrays
    train_bb_data = np.array(train_bb_data)
    train_bb_targets = np.array(train_bb_targets)
    val_bb_data = np.array(val_bb_data)
    val_bb_targets = np.array(val_bb_targets)

    print("Bounding Box Regression - Training data shape:", train_bb_data.shape)
    print("Bounding Box Regression - Training targets shape:", train_bb_targets.shape)
    print("Bounding Box Regression - Validation data shape:", val_bb_data.shape)
    print("Bounding Box Regression - Validation targets shape:", val_bb_targets.shape)

    # Define and compile the bounding box regression model
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

    # Train the BB regx model
    bb_history = bb_model.fit(train_bb_data, train_bb_targets, epochs= 500 , validation_data=(val_bb_data, val_bb_targets))

    # Evaluate the BB regx model
    bb_loss, bb_mae = bb_model.evaluate(val_bb_data, val_bb_targets)
    print(f"BB regx model evaluation - Loss: {bb_loss}, MAE: {bb_mae}")

    # Plot BB regx model training history
    plt.plot(bb_history.history['loss'], label='Train Loss')
    plt.plot(bb_history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Bounding Box Regression - Training and Validation Loss')

    plt.tight_layout()
    plt.show()

    # Fit BB regressor model
    bb_history = bb_model.fit(train_bb_data, train_bb_targets, epochs=50, validation_data=(val_bb_data, val_bb_targets))

    # Evaluate BB regressor model on training data
    train_loss, train_mae = bb_model.evaluate(train_bb_data, train_bb_targets)
    print(f"BB regressor model - Training MAE: {train_mae}")

    # Evaluate BB regressor model on validation data
    val_loss, val_mae = bb_model.evaluate(val_bb_data, val_bb_targets)
    print(f"BB regressor model - Validation MAE: {val_mae}")

    # Plot BB model training history
    plt.plot(bb_history.history['mae'], label='Train MAE')
    plt.plot(bb_history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('BB Regressor Model Training MAE')
    plt.show()

def softmax_classifier() :
    import os
    import json
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.models import Sequential
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

    # One-hot encode labels
    num_classes = 2  # Number of classes
    y = tf.keras.utils.to_categorical(y, num_classes)

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
    def create_model(input_shape=(224, 224, 3), num_classes=2):
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
            Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
        ])
        return model

    model = create_model(num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

    def plot_training_history(history):
        plt.figure(figsize=(12, 8))

        # Plot training and validation accuracy
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Plot training and validation loss
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    plot_training_history(history)



def parallel_execution():
    print("Starting the parallel execution of FCN pipeline")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(main_regression)  # Corrected here
        future2 = executor.submit(softmax_classifier)  # Corrected here
        result1 = future1.result()
        result2 = future2.result()
        print(result1)
        print(result2)
        print("Both pools completed successfully and execution is over")
    print("This is outside the parallel execution")

parallel_execution()
