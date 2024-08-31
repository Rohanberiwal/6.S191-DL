import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteHead
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import math 
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def extract_bounding_box(filename, path_to_file, mitotic_annotation_file, non_mitotic_annotation_file):
    if mitotic_annotation_file in path_to_file:
        annotation_file = mitotic_annotation_file
    elif non_mitotic_annotation_file in path_to_file:
        annotation_file = non_mitotic_annotation_file
    else:
        raise ValueError("File not found in either mitotic or non-mitotic annotation files.")

    with open(annotation_file) as f:
        annotations = json.load(f)

    if filename in annotations:
        annotation = annotations[filename]
        boxes = []
        for region in annotation['regions']:
            shape_attr = region['shape_attributes']
            x = shape_attr['x']
            y = shape_attr['y']
            width = shape_attr['width']
            height = shape_attr['height']
            boxes.append([x, y, x + width, y + height])
        return boxes
    else:
        raise ValueError(f"Filename '{filename}' not found in '{annotation_file}'.")


def print_mitotic(json_mitotic):
    print("This is the mitotic printer function")
    standard_dict_mitotic = {}
    universal_list = []
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', 'N/A')
            y = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height

            print(f"Bounding Box Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            universal_list.append([x_min, y_min, x_max, y_max])
        
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []

    return standard_dict_mitotic

def print_filename_bbox(json_file):
    print("This is the printer filename function for non-mitotic")
    standard_dict_non_mitotic = {}
    universal_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', 'N/A')
            y = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height

            print(f"Bounding Box Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            universal_list.append([x_min, y_min, x_max, y_max])
        
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []

    return standard_dict_non_mitotic


def extract_filenames_from_json(json_file, root):
    with open(json_file) as f:
        data = json.load(f)
    filename_list = []
    for filename, attributes in data.items():
        filename = filename.replace('.jpeg', '.jpg')
        img_name = attributes['filename'] 
        img_path = os.path.join(root, img_name)  
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


root = r'/content/train_mitotic/Train_mitotic'
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
print(standard_dict_mitotic)
print(standard_dict_non_mitotic)
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [0, x_center, y_center, width, height]  
    
def plot_bounding_boxes(standard_dict, root_dir, output_img_dir, output_lbl_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_lbl_dir):
        os.makedirs(output_lbl_dir)
    
    for img_path, bboxes in standard_dict.items():
        img = Image.open(img_path)
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)
        
        yolo_labels = []
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
            yolo_label = convert_bbox_to_yolo_format(bbox, img_width, img_height)
            yolo_labels.append(yolo_label)
        

        img_filename = os.path.basename(img_path)
        img.save(os.path.join(output_img_dir, img_filename))

        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        with open(os.path.join(output_lbl_dir, txt_filename), 'w') as f:
            for label in yolo_labels:
                f.write(" ".join(map(str, label)) + '\n')

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

output_img_dir = '/content/yolov8/images'
output_lbl_dir = '/content/yolov8/labels'
plot_bounding_boxes(standard_dict_mitotic, root, output_img_dir, output_lbl_dir)

import os
import shutil
import random

def split_dataset(img_dir, lbl_dir, output_train_img_dir, output_train_lbl_dir, output_val_img_dir, output_val_lbl_dir, split_ratio=0.3):
    # Create directories if they don't exist
    os.makedirs(output_train_img_dir, exist_ok=True)
    os.makedirs(output_train_lbl_dir, exist_ok=True)
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_lbl_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.shuffle(all_files)  
    split_index = int(len(all_files) * (1 - split_ratio))  
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
    
    for file_name in train_files:
        shutil.copy(os.path.join(img_dir, file_name), os.path.join(output_train_img_dir, file_name))
        shutil.copy(os.path.join(lbl_dir, file_name.replace('.jpg', '.txt')), os.path.join(output_train_lbl_dir, file_name.replace('.jpg', '.txt')))
    
    for file_name in val_files:
        shutil.copy(os.path.join(img_dir, file_name), os.path.join(output_val_img_dir, file_name))
        shutil.copy(os.path.join(lbl_dir, file_name.replace('.jpg', '.txt')), os.path.join(output_val_lbl_dir, file_name.replace('.jpg', '.txt')))

    print(f"Dataset split into {len(train_files)} training and {len(val_files)} validation images.")


split_dataset(
    img_dir='/content/yolov8/images',
    lbl_dir='/content/yolov8/labels',
    output_train_img_dir='/content/yolov8/train/images',
    output_train_lbl_dir='/content/yolov8/train/labels',
    output_val_img_dir='/content/yolov8/val/images',
    output_val_lbl_dir='/content/yolov8/val/labels'
)
# Create the YAML configuration file
data_yaml = """
path: /content/yolov8  # Root directory
train: train/images  # Path to training images
val: val/images  # Path to validation images

nc: 1  # Number of classes
names: ['mitotic']  # Class names
"""

# Write the YAML file to the specified path
yaml_path = '/content/yolov8/data.yaml'
with open(yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"YAML file saved to {yaml_path}")

from ultralytics import YOLO
import os

# Define paths
yaml_path = '/content/yolov8/data.yaml'
train_img_dir = '/content/yolov8/train/images'
train_lbl_dir = '/content/yolov8/train/labels'
val_img_dir = '/content/yolov8/val/images'
val_lbl_dir = '/content/yolov8/val/labels'

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or use a different pre-trained model if desired
class CustomCallback:
    def __init__(self):
        self.epoch_losses = []
        self.epoch_val_losses = []
        self.epochs = []
        self.best_epoch = None
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.epoch_losses.append(loss)
        self.epoch_val_losses.append(val_loss)
        self.epochs.append(epoch)
        print(f"Epoch {epoch}: Loss = {loss}, Val Loss = {val_loss}")
        
        # Track the best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            print(f"New best model found at epoch {epoch} with Val Loss = {val_loss}")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.epoch_losses, label='Training Loss')
        plt.plot(self.epochs, self.epoch_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.savefig('/content/yolov8/loss_plot.png')
        plt.show()


model = YOLO('yolov8n.pt')  # or use a different pre-trained model if desired

from ultralytics import YOLO
import matplotlib.pyplot as plt

data_yaml = """
path: /content/yolov8  # Root directory
train: train/images  # Path to training images
val: val/images  # Path to validation images

nc: 1  # Number of classes
names: ['mitotic']  # Class names
"""

# Write the YAML file to the specified path
yaml_path = '/content/yolov8/data.yaml'
with open(yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"YAML file saved to {yaml_path}")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or use a different pre-trained model if desired

training_config = {
    'data': yaml_path,
    'epochs': 50,  # number of training epochs
    'batch': 16,  # batch size
    'imgsz': 640,  # image size
    'optimizer': 'Adam',  # optimizer
    'lr0': 0.01,  # initial learning rate
    'momentum': 0.937,  # momentum
    'weight_decay': 0.0005,  # weight decay
    'val': True,  # Ensure validation is used
    'val_batch_size': 16  # Batch size for validation
}

print(results)
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def validate_model(model, val_img_dir, val_lbl_dir):
    from ultralytics import YOLO
    import os
    import cv2
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # Define paths
    image_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    label_paths = [os.path.join(val_lbl_dir, f.replace('.jpg', '.txt')) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]

    # Check if the number of images matches the number of labels
    if len(image_paths) != len(label_paths):
        print("Mismatch between number of images and labels.")
        return

    # Validate images and labels
    for img_path, lbl_path in zip(image_paths, label_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        
        height, width, _ = img.shape
        if width <= 0 or height <= 0:
            print(f"Invalid image dimensions for: {img_path}")
            continue
        
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as file:
                for line in file:
                    labels.append(line.strip().split())
        else:
            print(f"No labels found for image: {img_path}")
            continue
        
        # Your validation logic here
        # For example, running inference on the image
        results = model(img)
        
        # Plot results
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Results for {os.path.basename(img_path)}")
        plt.show()
        
        print(f"Validated {img_path}")

validate_model(model, val_img_dir, val_lbl_dir)
print("every code ends for good")
