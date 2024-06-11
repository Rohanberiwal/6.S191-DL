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

print(tf.__version__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'

# Selective search function
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
    
    
def classify_feature_vector(feature_vector, svm_model):
    feature_vector_np = np.array(feature_vector).reshape(1, -1)
    prediction = svm_model.predict(feature_vector_np)
    return prediction

# Get ground truth bounding boxes
bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******\n")

# Get selective search proposals
print("****** SelectiVE search ********\n")
proposals = get_proposals(image_path)
print("The proposal regions are:", proposals)
print("Number of Region Proposals:", len(proposals))
print("\n****** SelectiVE search ********\n")

# Extract and plot features using VGG16
print("****** CNN Layer ********\n")
features = extract_features(image_path)
print(features)
plot_feature_maps(features)
print("Shape of feature map:", features.shape)
print("\n****** CNN Layer ********\n")

# Load image with OpenCV
image = cv2.imread(image_path)
height, width = image.shape[:2]
print("Height:", height)
print("Width:", width)

# Load the pre-trained FCN32 model from PyTorch Hub
fcn32_model = fcn_resnet50(pretrained=True)
fcn32_model = fcn32_model.to(device)
fcn32_model.eval()

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.CenterCrop(224),      
    transforms.ToTensor(),           
    transforms.Normalize(            
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess the image
image = Image.open(image_path)
input_img = transform(image).unsqueeze(0).to(device)  

# Run the FCN model and get the output
with torch.no_grad():
    output = fcn32_model(input_img)['out']
    print("running ....")
print(output)
print("\n")

# Ensure features from VGG16 are in a compatible format for PyTorch
features = torch.tensor(features).permute(0, 3, 1, 2).to(device)
pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
feature_vector = pooled_features.view(pooled_features.size(0), -1)
normalized_feature_vector = F.normalize(feature_vector, p=2, dim=1)

print(normalized_feature_vector)


feature_maps = np.array(features)
feature_vectors = np.array(feature_vector)
labels = [0,1]
# Flatten the feature maps
flattened_feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)

# Combine the flattened feature maps and feature vectors
combined_features = np.hstack((flattened_feature_maps, feature_vectors))

# Train the SVM model
svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(combined_features,labels )

# Save the trained SVM model
joblib.dump(svm_model, 'trained_svm_model.pkl')

# Load the trained SVM model
svm_model = joblib.load('trained_svm_model.pkl')

# Predict using the SVM model
def classify_features(features, svm_model):
    # Flatten the feature maps
    flattened_feature_maps = features['feature_maps'].reshape(features['feature_maps'].shape[0], -1)
    
    # Combine the flattened feature maps and feature vectors
    combined_features = np.hstack((flattened_feature_maps, features['feature_vectors']))
    
    # Predict using the SVM model
    prediction = svm_model.predict(combined_features)
    return prediction

# Example usage
test_features = {'feature_maps': features , 'feature_vectors':feature_vector}
prediction = classify_features(test_features, svm_model)
# Check the lengths of combined_features and labels
print("Number of samples in combined_features:", len(combined_features))
print("Number of labels:", len(labels))

print("Prediction:", prediction)

