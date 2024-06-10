import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn import svm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
import pandas as pd 
import matplotlib.pyplot as plt

import json
import os 
print(tf.__version__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"

import json


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

def  ground_truth_func(json_file):
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


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to fit VGG16 input size
    image = preprocess_input(image)  # Preprocess input according to VGG16 requirements
    image = image.reshape(1, *image.shape)  # Reshape to match VGG16 input shape
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



def build_fcn(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Decoder
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(pool2)
    concat1 = tf.keras.layers.Concatenate()([up1, conv2])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat2 = tf.keras.layers.Concatenate()([up2, conv1])
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(concat2)
    conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv8)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


json_file_path = 'A00.json'
bounding_boxes = ground_truth_func(json_file_path)


print("\n")
print("********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******")
print("\n")


print("****** SelectiVE search ********")
print("\n")
print("Selective search ROI ") 
proposals = get_proposals(image_path)
print("The proposal region are ",proposals)
print("Number of Region Proposals:", len(proposals))
print("\n")
print("****** SelectiVE search ********")


print("****** CNN Layer ********")
print("\n")
features = extract_features(image_path)
print(features)
plot_feature_maps(features)
print("Shape of feature map:", features.shape)
print("\n")
print("****** CNN Layer ********")



