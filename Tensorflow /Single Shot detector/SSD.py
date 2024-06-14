import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import torch
import torchvision.models as models
from sklearn import svm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd 
import torch.nn as nn
import matplotlib.pyplot as plt
import deepchem as dp 
import json
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Softmax
from tensorflow.keras.models import Model
import tarfile
print(tf.__version__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
tf.get_logger().setLevel('ERROR')
tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(True)
tf.autograph.set_verbosity(0)

global_image_file_paths = []
svm_model = svm.SVC(kernel='linear')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
global_image_features = {}
tar_file_path = r"C:\Users\rohan\OneDrive\Desktop\datasets"


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
                bounding_box = [x, y, width, height]
                bounding_boxes.append(bounding_box)
    return bounding_boxes

# VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  
    image = preprocess_input(image)  
    image = np.expand_dims(image, axis=0)  
    features = base_model.predict(image)
    return features


def generate_anchor_boxes(feature_map_shape, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.

    Args:
    - feature_map_shape (tuple): Shape of the feature map (height, width).
    - scales (list): List of scales (sizes) of the anchor boxes.
    - aspect_ratios (list): List of aspect ratios of the anchor boxes.

    Returns:
    - anchor_boxes (numpy array): Array of anchor boxes in the format (num_boxes, 4).
      Each row represents [x_min, y_min, x_max, y_max] of an anchor box.
    """
    anchor_boxes = []
    feature_map_height, feature_map_width = feature_map_shape

    # Iterate over each cell in the feature map
    for i in range(feature_map_height):
        for j in range(feature_map_width):
            # Compute center of the current cell
            center_x = (j + 0.5) / feature_map_width
            center_y = (i + 0.5) / feature_map_height

            # Generate anchor boxes for each scale and aspect ratio
            for scale in scales:
                for ratio in aspect_ratios:
                    # Compute width and height of the anchor box
                    width = scale * np.sqrt(ratio)
                    height = scale / np.sqrt(ratio)

                    # Calculate coordinates of the anchor box
                    x_min = center_x - width / 2
                    y_min = center_y - height / 2
                    x_max = center_x + width / 2
                    y_max = center_y + height / 2

                    # Append to anchor_boxes list
                    anchor_boxes.append([x_min, y_min, x_max, y_max])

    return np.array(anchor_boxes)



def predict_class_scores(base_model_output, num_classes, num_boxes_per_cell):
    # Example: Convolutional layer for predicting class scores
    cls = Conv2D(num_boxes_per_cell * num_classes, kernel_size=(3, 3), padding='same', activation='linear')(base_model_output)
    class_scores = Reshape((-1, num_classes))(cls)
    class_probs = Softmax(axis=-1)(class_scores)
    print(class_scores)
    return class_probs

 


def visualize_anchor_boxes(image_path, anchor_boxes):
    """
    Visualize anchor boxes overlaid on the image.
    
    Args:
    - image_path: Path to the input image.
    - anchor_boxes: List of anchor boxes in the format [y_min, x_min, y_max, x_max].
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (matplotlib expects RGB format)
    
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot each anchor box
    for box in anchor_boxes:
        #print(box)
        y_min, x_min, y_max, x_max = box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                          fill=False, edgecolor='green', linewidth=2))
    
    plt.axis('off')
    plt.show()


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters:
    - boxes: Nx4 numpy array of bounding boxes.
    - scores: N numpy array of scores.
    - iou_threshold: IoU threshold for filtering boxes.

    Returns:
    - indices: List of indices of selected boxes.
    """
    # Sort boxes by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while sorted_indices.size > 0:
        current_idx = sorted_indices[0]
        selected_indices.append(current_idx)
        remaining_indices = sorted_indices[1:]

        to_remove = []
        for idx in remaining_indices:
            if iou(boxes[current_idx], boxes[idx]) > iou_threshold:
                to_remove.append(idx)
        
        sorted_indices = np.delete(sorted_indices, to_remove)

    return selected_indices



# Get ground truth bounding boxes
bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******\n")


# Extract and plot features using VGG16
print("****** CNN Layer ********\n")
features = extract_features(image_path)
print(features)
print("Shape of feature map:", features.shape)
print("\n****** CNN Layer ********\n")



print("******** anchor box ********")
feature_map_shape = (features.shape[1], features.shape[2])  
scales = [0.5, 1.0, 2.0]
aspect_ratios = [0.5, 1.0, 2.0]
"""
Why hard coded this  ?
0.5 represents smaller boxes relative to the objects,
1.0 represents boxes approximately the same size as the objects,
2.0 represents larger boxes relative to the objects.

"""

anchor_boxes = generate_anchor_boxes(feature_map_shape, scales, aspect_ratios)
print("Generated Anchor Boxes:\n", anchor_boxes)
print("Number of Anchor Boxes:", len(anchor_boxes))
visualize_anchor_boxes(image_path, anchor_boxes)

"""

base_model_output = base_model.output
num_classes = 2
num_boxes_per_cell = 9

# Add custom layers for class predictions
base_model_output = base_model.output
class_probs = predict_class_scores(base_model_output, num_classes, num_boxes_per_cell)
model = Model(inputs=base_model.input, outputs=class_probs)

# Predict class probabilities for each anchor box
for i, anchor_box in enumerate(anchor_boxes):
    # Prepare input for the anchor box
    x_min, y_min, x_max, y_max = anchor_box
    dummy_input = np.random.random((1, 224, 224, 3))  # Replace with actual image data for each anchor box
    class_probs_array = model.predict(dummy_input)
    print(f"Anchor Box {i+1}:")
    print("Class probabilities array shape:", class_probs_array.shape)
    print(class_probs_array)
    print()

"""
base_model_output = base_model.output
num_classes = 2
num_boxes_per_cell = 9

# Add custom layers for class predictions
class_probs = predict_class_scores(base_model_output, num_classes, num_boxes_per_cell)
model = Model(inputs=base_model.input, outputs=class_probs)

# List to store anchor box predictions
anchor_box_predictions = []

# Predict class probabilities for each anchor box
for i, anchor_box in enumerate(anchor_boxes):
    # Prepare input for the anchor box
    x_min, y_min, x_max, y_max = anchor_box
    dummy_input = np.random.random((1, 224, 224, 3))  # Replace with actual image data for each anchor box
    class_probs_array = model.predict(dummy_input)
    
    # Store anchor box index (or name) and class probabilities
    anchor_box_prediction = {
        'anchor_box_index': i,
        'confidence_scores': class_probs_array.tolist()  # Convert to list for easier handling
    }
    
    anchor_box_predictions.append(anchor_box_prediction)

    # Print anchor box predictions (optional)
    print(f"Anchor Box {i+1}:")
    print("Class probabilities array shape:", class_probs_array.shape)
    print(class_probs_array)
    print()

for prediction in anchor_box_predictions:
    print(f"Anchor Box Index: {prediction['anchor_box_index']}")
    print("Confidence Scores:")
    print(prediction['confidence_scores'])
    print()

print("anchor box coming over ")
print(anchor_box_predictions)

    
"""
    
index_output = non_max_suppression(anchor_box , anchor_box_predictions, iou_threshold=0.5)
print(index_output)
"""
