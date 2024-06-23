import torch
import torch.nn as nn
import torchvision.ops as ops
import cv2
import numpy as np
import json
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

# Define the ROI pooling layer class
class ROIPoolingLayer(nn.Module):
    def __init__(self, output_size):
        super(ROIPoolingLayer, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, proposals):
        return ops.roi_pool(feature_map, proposals, self.output_size)

# Function for selective search
def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

# Function to get proposals
def get_proposals(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    proposals = selective_search(image)
    formatted_proposals = []
    for x, y, w, h in proposals:
        x_max = x + w
        y_max = y + h
        # Convert to [batch_index, x_min, y_min, x_max, y_max]
        formatted_proposals.append([0, x, y, x_max, y_max])
    
    return formatted_proposals

# Function to extract features using VGG16
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = base_model.predict(image)
    return features

# Function to plot feature maps
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

# Main execution
if __name__ == "__main__":
    try:
        # Initialize VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Get the path to the image and JSON file
        image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpeg"
        json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'

        # Get ground truth bounding boxes
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
        plot_feature_maps(features)
        print("Shape of feature map:", features.shape)
        print("\n****** CNN Layer ********\n")

        # Convert proposals to tensor
        proposals = torch.tensor(proposals).float()

        # Initialize ROI Pooling layer
        roi_output_size = (7, 7)  # Example output size
        roi_pooling_layer = ROIPoolingLayer(output_size=roi_output_size)

        # Convert features to torch tensor and change shape to (N, C, H, W)
        features = torch.tensor(features).permute(0, 3, 1, 2)

        # Apply ROI Pooling
        pooled_features = roi_pooling_layer(features, proposals)
        print("Shape of pooled features:", pooled_features.shape)

        # Flatten pooled features
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        print("Shape of flattened features:", flattened_features.shape)
        print("Flattened features:", flattened_features)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
