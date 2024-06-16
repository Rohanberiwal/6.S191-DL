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

class ROIPoolingLayer(nn.Module):
    def __init__(self, output_size):
        super(ROIPoolingLayer, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, proposals):
        return ops.roi_pool(feature_map, proposals, self.output_size)

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
                bounding_box = [x, y, x + width, y + height]  # Convert to [x_min, y_min, x_max, y_max]
                bounding_boxes.append(bounding_box)
    return bounding_boxes

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = base_model.predict(image)
    return features

def plot_feature_maps(features):
    num_maps = features.shape[-1]
    num_cols = 8
    num_rows = num_maps // num_cols + 1
    plt.figure(figsize=(20, 20))
    for i in range(num_maps):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(features[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Feature Map {i + 1}")
    plt.show()


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'

# Get ground truth bounding boxes
bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')
print("********Actual Ground Truth of the Bounding Boxes******\n")

# Get selective search proposals
print("****** Selective Search ********\n")
proposals = get_proposals(image_path)
print("The proposal regions are:", proposals)
print("Number of Region Proposals:", len(proposals))
print("\n****** Selective Search ********\n")

# Extract and plot features using VGG16
print("****** CNN Layer ********\n")
features = extract_features(image_path)
print(features)
plot_feature_maps(features)
print("Shape of feature map:", features.shape)
print("\n****** CNN Layer ********\n")


# Get proposals
proposals = get_proposals(image_path)

# Format proposals as [batch_index, x_min, y_min, x_max, y_max]
proposals = [[0, x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max in proposals]

# Convert proposals to tensor
proposals = torch.tensor(proposals).float()

# Initialize ROI Pooling layer
roi_output_size = (7, 7)  # Example output size
roi_pooling_layer = ROIPoolingLayer(output_size=roi_output_size)

# Convert features to torch tensor and change shape to (N, C, H, W)
features = torch.tensor(features).permute(0, 3, 1, 2)

# Apply ROI Pooling
pooled_features = roi_pooling_layer(features, proposals)
print(pooled_features.shape)  

