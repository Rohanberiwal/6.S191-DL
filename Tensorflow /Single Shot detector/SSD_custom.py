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
from tensorflow.keras.applications import ResNet101 
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
import torch

import tensorflow as tf
import numpy as np

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

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches
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
from tensorflow.keras.applications import ResNet101 
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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches

root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"


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

            boundary_box.append([xmin, ymin, width, height])  
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box 

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


def plot_boxes_together(dict_mitotic, dict_non_mitotic):
    common_keys = set(dict_mitotic.keys()).intersection(dict_non_mitotic.keys())

    for key in common_keys:
        try:
            # Load the image
            image = Image.open(key)
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            ax = plt.gca()


            for xmin, ymin, width, height in dict_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))  # Resize for display
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
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

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths



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



def get_output_feature_maps(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    intermediate_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    output_feature_maps = intermediate_model.predict(image)
    output_feature_maps_np = [np.array(fmap) for fmap in output_feature_maps]
    return output_feature_maps_np

def create_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
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

def plot_anchor_boxes(image_path, anchor_boxes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    resized_image = cv2.resize(image, (224, 224))

    plt.figure(figsize=(10, 8))
    plt.imshow(resized_image)
    ax = plt.gca()

    # Plot anchor boxes
    for box in anchor_boxes:
        xmin = int(box[0] * 224)  # Scale coordinates to image size
        ymin = int(box[1] * 224)
        xmax = int(box[2] * 224)
        ymax = int(box[3] * 224)
        
        # Draw rectangle
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.title('Input Image with Anchor Boxes (in Green)')
    plt.axis('off')
    plt.show()
    

print_mitotic(mitotic_annotation_file)
print_filename_bbox(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

print("Patches saved successfully.")
list_mitotic = get_file_paths(mitotic_save_dir)
list_non_mitotic = get_file_paths(non_mitotic_save_dir)
print("List of mitotic patches:", list_mitotic)
print("List of non-mitotic patches:", list_non_mitotic)
labels_mitotic = [1] * len(list_mitotic)
labels_non_mitotic = [0] * len(list_non_mitotic)
X = np.array(list_mitotic + list_non_mitotic)
y = np.array(labels_mitotic + labels_non_mitotic)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]


split_idx = int(0.7 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


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

plot_training_history(history)
image_path  = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
features = []
featured_map = get_output_feature_maps(model, image_path)

print(featured_map)
    
print("Pipelien ends")


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
print("CNN is tuned")
print("Starting the code")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Force TensorFlow to use GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def remove_empty_feature_maps(features):
    non_empty_indices = []
    for idx, feature_map in enumerate(features):
        if not np.all(feature_map == 0):
            non_empty_indices.append(idx)
    filtered_features = [features[idx] for idx in non_empty_indices]
    return filtered_features, non_empty_indices

print("This are the globaal dicts")
print(standard_dict_mitotic)
print("\n")
print(standard_dict_non_mitotic)
print("\n")
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
features = get_output_feature_maps(model, image_path)
print(features)
print("\n****** CNN Layer ********\n")
print("This is the code  for the ROI pooling network")

print("tHIS IS THE TESTIFN SEGEMNET")
for i in features  :
    print(np.array(i))

print("This is the code for the matric pooling")

import torch
import torch.nn.functional as F

def crop_and_resize_torch(image_tensor, bbox, target_size):
    """
    Crop and resize a region of interest (ROI) from an image tensor (PyTorch).

    Parameters:
    - image_tensor (torch.Tensor): The input image tensor.
    - bbox (list): List containing bounding box coordinates [xmin, ymin, xmax, ymax].
    - target_size (tuple): Target size (height, width) for resizing.

    Returns:
    - resized_roi (torch.Tensor): Resized ROI as a PyTorch tensor.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Ensure integer values
    roi = image_tensor[:, :, ymin:ymax, xmin:xmax]  # Ensure correct channel dimension
    resized_roi = F.interpolate(roi, size=target_size, mode='nearest')
    return resized_roi.squeeze(0)



def convert_to_normalized_boxes(proposals, image_height, image_width):
    """
    Convert bounding box proposals to normalized coordinates.

    Parameters:
    - proposals (numpy.ndarray): Array of bounding box proposals in format [y_min, x_min, height, width].
    - image_height (int): Height of the input image or feature map.
    - image_width (int): Width of the input image or feature map.

    Returns:
    - normalized_boxes (numpy.ndarray): Array of normalized bounding boxes in format [y_min, x_min, y_max, x_max].
    """
    normalized_boxes = np.zeros_like(proposals, dtype=np.float32)

    for i, proposal in enumerate(proposals):
        y_min, x_min, height, width = proposal
        y_max = y_min + height
        x_max = x_min + width

        # Normalize coordinates
        normalized_boxes[i, 0] = y_min / image_height
        normalized_boxes[i, 1] = x_min / image_width
        normalized_boxes[i, 2] = y_max / image_height
        normalized_boxes[i, 3] = x_max / image_width

    return normalized_boxes

def roi_align(feature_maps, boxes, output_height, output_width):
    """
    `feature_maps` is a list of 2-D arrays, each representing an input feature map
    `boxes` is a list of lists, where each inner list represents a bounding box [x, y, w, h]
    `output_height` and `output_width` are the desired spatial size of output feature map
    """
    num_feature_maps = len(feature_maps)
    num_boxes = len(boxes)
    output_feature_maps = []

    for fmap_idx in range(num_feature_maps):
        image = feature_maps[fmap_idx]
        img_height, img_width = image.shape
        fmap_boxes = boxes[fmap_idx]
        fmap_feature_maps = []

        for box in fmap_boxes:
            x, y, w, h = box
            feature_map = []

            for i in range(output_height):
                for j in range(output_width):
                    # Calculate coordinates in the original image space
                    y_orig = y + i * (h / output_height)
                    x_orig = x + j * (w / output_width)

                    y_l = int(np.floor(y_orig))
                    y_h = int(np.ceil(y_orig))
                    x_l = int(np.floor(x_orig))
                    x_h = int(np.ceil(x_orig))

                    # Clip indices to stay within image bounds
                    y_l = np.clip(y_l, 0, img_height - 1)
                    y_h = np.clip(y_h, 0, img_height - 1)
                    x_l = np.clip(x_l, 0, img_width - 1)
                    x_h = np.clip(x_h, 0, img_width - 1)

                    a = image[y_l, x_l]
                    b = image[y_l, x_h]
                    c = image[y_h, x_l]
                    d = image[y_h, x_h]

                    y_weight = y_orig - y_l
                    x_weight = x_orig - x_l

                    val = a * (1 - x_weight) * (1 - y_weight) + \
                          b * x_weight * (1 - y_weight) + \
                          c * y_weight * (1 - x_weight) + \
                          d * x_weight * y_weight

                    feature_map.append(val)

            fmap_feature_maps.append(np.array(feature_map).reshape(output_height, output_width))

        output_feature_maps.append(fmap_feature_maps)

    return output_feature_maps


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
def process_roi_proposals(proposals, feature_maps):
    num_feature_maps = feature_maps.shape[-1]
    
    for fmap_idx in range(feature_maps.shape[0]):
        fmap = feature_maps[fmap_idx]
        
        print(f"Processing Feature Map {fmap_idx+1} with shape {fmap.shape}")
        
        for i, proposal in enumerate(proposals):
            x, y, w, h = proposal
            x_end = x + w
            y_end = y + h
            
            # Extract ROI from feature map
            roi_feature_maps = fmap[y:y_end, x:x_end, :]
            print(f"Proposal {i+1}: {proposal}")
            print(f"Extracted ROI shape: {roi_feature_maps.shape}")
            
            # Normalize coordinates for crop_and_resize
            img_h, img_w, _ = fmap.shape
            boxes = np.array([[y/img_h, x/img_w, y_end/img_h, x_end/img_w]], dtype=np.float32)
            box_indices = np.zeros(1, dtype=np.int32)
            
            # Crop and resize ROI
            roi = tf.image.crop_and_resize(fmap[np.newaxis], boxes, box_indices, [7, 7])
            print(f"ROI shape after crop_and_resize: {roi.shape}")
            
            # Apply Global Pooling
            avg_pool = GlobalAveragePooling2D()(roi)
            max_pool = GlobalMaxPooling2D()(roi)
            
            print(f"Feature map {fmap_idx+1}, Proposal {i+1}:")
            print(f"Number of CNN feature maps extracted: {num_feature_maps}")
            print(f"Shape of pooled feature maps (Average Pooling): {avg_pool.shape}")
            print(f"Shape of pooled feature maps (Max Pooling): {max_pool.shape}")
            print(f"Feature vector (Max Pooling): {max_pool.numpy().flatten()}")
            print()


base_model = ResNet101(weights='imagenet', include_top=False)
feature_model = Model(inputs=base_model.input, outputs=base_model.output)


































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
def plot_predicted_boxes(model, images, targets):
    try:
        plt.figure(figsize=(15, 10))
        for i in range(len(images)):
            image = images[i]
            target = targets[i]

            # Reshape the image for prediction
            image = image.reshape((1,) + image.shape)

            # Predict bounding box coordinates
            prediction = model.predict(image)
            xmin_pred = prediction[0][0] * image.shape[2]  # Scale predictions back to image dimensions
            ymin_pred = prediction[0][1] * image.shape[1]
            width_pred = prediction[0][2] * image.shape[2]
            height_pred = prediction[0][3] * image.shape[1]

            # Plot original image
            plt.subplot(2, 3, i + 1)
            plt.imshow(image[0])
            plt.title(f"Image {i + 1}")

            # Plot ground truth bounding box
            xmin_true = target[0] * image.shape[2]
            ymin_true = target[1] * image.shape[1]
            width_true = target[2] * image.shape[2]
            height_true = target[3] * image.shape[1]
            rect_true = patches.Rectangle((xmin_true, ymin_true), width_true, height_true, linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect_true)

            # Plot predicted bounding box
            rect_pred = patches.Rectangle((xmin_pred, ymin_pred), width_pred, height_pred, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect_pred)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting predicted boxes: {e}")

# Function to test bounding box adjustments and plot on image
def test_and_plot_bbox_adjustment(model, image_path, predicted_bbox):
    try:
        # Load and preprocess the image
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict bounding box coordinates
        prediction = model.predict(image_array)
        xmin_pred = prediction[0][0] * image.width  # Scale predictions back to image dimensions
        ymin_pred = prediction[0][1] * image.height
        width_pred = prediction[0][2] * image.width
        height_pred = prediction[0][3] * image.height

        # Adjust the predicted bounding box (e.g., use predicted_bbox to adjust)
        xmin_adjusted = xmin_pred + predicted_bbox[0]
        ymin_adjusted = ymin_pred + predicted_bbox[1]
        width_adjusted = width_pred + predicted_bbox[2]
        height_adjusted = height_pred + predicted_bbox[3]

        # Plot the image with adjusted bounding box
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        ax = plt.gca()

        # Plot predicted bounding box
        rect_pred = patches.Rectangle((xmin_adjusted, ymin_adjusted), width_adjusted, height_adjusted, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_pred)

        plt.title('Image with Adjusted Bounding Box')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error testing and plotting bounding box adjustment: {e}")


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
    test_and_plot_bbox_adjustment(bb_model, image_path, predicted_bbox = [])
    
    
