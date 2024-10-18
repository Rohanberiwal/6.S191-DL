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
import os
import zipfile
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
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
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import zipfile
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
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

def extract_zip_file(input_zip_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Files extracted to {output_dir}")


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


import os
import shutil
import random

import os
import random
import shutil
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms.functional as F


def rotate_bbox(bbox, angle, img_width, img_height):
    class_id, x_center, y_center, width, height = bbox

    # Convert center-based bbox to corner points (top-left and bottom-right)
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)

    # Calculate the center of the image
    cx, cy = img_width / 2, img_height / 2

    # Rotate points
    def rotate_point(x, y, angle_rad, cx, cy):
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        x_new = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
        y_new = sin_angle * (x - cx) + cos_angle * (y - cy) + cy
        return x_new, y_new

    # Rotate each corner
    angle_rad = torch.tensor(angle * (3.141592653589793 / 180.0))  # Convert to radians
    x_min_new, y_min_new = rotate_point(x_min, y_min, angle_rad, cx, cy)
    x_max_new, y_max_new = rotate_point(x_max, y_max, angle_rad, cx, cy)

    # Convert back to center format
    new_x_center = (x_min_new + x_max_new) / 2
    new_y_center = (y_min_new + y_max_new) / 2
    new_width = x_max_new - x_min_new
    new_height = y_max_new - y_min_new

    return [class_id, new_x_center, new_y_center, new_width, new_height]




def augment_image_and_bbox(image, bbox, img_path):
    augmentations = [
        {'rotation': 90},   # Rotation by 90 degrees
        {'rotation': 180},  # Rotation by 180 degrees
        {'rotation': 270},  # Rotation by 270 degrees
        {'brightness': 1.5} # Increase brightness by a factor of 1.5
    ]

    img_width, img_height = image.size
    augmented_images = []
    augmented_bboxes = []

    for aug in augmentations:
        if 'rotation' in aug:
            # Rotate image and adjust bounding box
            angle = aug['rotation']
            rotated_img = image.rotate(angle, expand=True)
            rotated_bbox = rotate_bbox(bbox, angle, img_width, img_height)
            augmented_images.append(rotated_img)
            augmented_bboxes.append(rotated_bbox)
        elif 'brightness' in aug:
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(image)
            bright_img = enhancer.enhance(aug['brightness'])
            augmented_images.append(bright_img)
            augmented_bboxes.append(bbox)  # Bounding box remains the same for brightness adjustment

    return augmented_images, augmented_bboxes

def split_dataset_with_augmentation(img_dir, lbl_dir,
                                    output_train_img_dir, output_train_lbl_dir,
                                    output_val_img_dir, output_val_lbl_dir,
                                    output_aug_img_dir, output_aug_lbl_dir,
                                    split_ratio=0.3, augment=False):

    os.makedirs(output_train_img_dir, exist_ok=True)
    os.makedirs(output_train_lbl_dir, exist_ok=True)
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_lbl_dir, exist_ok=True)
    os.makedirs(output_aug_img_dir, exist_ok=True)
    os.makedirs(output_aug_lbl_dir, exist_ok=True)

    all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.shuffle(all_files)
    split_index = int(len(all_files) * (1 - split_ratio))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    for file_name in train_files:
        # Copy original training files
        img_path = os.path.join(img_dir, file_name)
        lbl_path = os.path.join(lbl_dir, file_name.replace('.jpg', '.txt'))
        shutil.copy(img_path, os.path.join(output_train_img_dir, file_name))
        shutil.copy(lbl_path, os.path.join(output_train_lbl_dir, file_name.replace('.jpg', '.txt')))

        # Perform augmentation if augment flag is set
        if augment:
            img = Image.open(img_path)
            with open(lbl_path, 'r') as f:
                bbox = list(map(float, f.readline().split()))  # Assuming the label format is [x, y, w, h]

            augmented_images, augmented_bboxes = augment_image_and_bbox(img, bbox, img_path)

            for i, (aug_img, aug_bbox) in enumerate(zip(augmented_images, augmented_bboxes)):
                aug_file_name = f"{os.path.splitext(file_name)[0]}_aug{i+1}.jpg"
                aug_img.save(os.path.join(output_aug_img_dir, aug_file_name))
                # Save the augmented bounding box
                with open(os.path.join(output_aug_lbl_dir, aug_file_name.replace('.jpg', '.txt')), 'w') as f:
                    f.write(f"{aug_bbox[0]} {aug_bbox[1]} {aug_bbox[2]} {aug_bbox[3]}\n")

    for file_name in val_files:

        shutil.copy(os.path.join(img_dir, file_name), os.path.join(output_val_img_dir, file_name))
        shutil.copy(os.path.join(lbl_dir, file_name.replace('.jpg', '.txt')), os.path.join(output_val_lbl_dir, file_name.replace('.jpg', '.txt')))

    print(f"Dataset split into {len(train_files)} training and {len(val_files)} validation images.")
    if augment:
        print(f"Augmented data saved with rotations and brightness adjustments.")

def validate_model(model, val_img_dir, val_lbl_dir):
    from ultralytics import YOLO
    import os
    import cv2
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    image_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    label_paths = [os.path.join(val_lbl_dir, f.replace('.jpg', '.txt')) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
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

        results = model(img)

        # Plot results
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Results for {os.path.basename(img_path)}")
        plt.show()

        print(f"Validated {img_path}")


def evaluate_model(model, test_img_dir):
    image_paths = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.jpg')]

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        # Run inference on the image
        results = model(img)

        # Accessing the predicted boxes, labels, and scores
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            # Print detection results
            print(f"Image: {os.path.basename(img_path)}")
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                print(f"Class: {cls_id}, Confidence: {conf}, Box: {box}")

            # Plot the image with detections
            result_img = result.plot()  # Annotated image
            plt.imshow(result_img)
            plt.axis('off')
            plt.title(f"Results for {os.path.basename(img_path)}")
            plt.show()

def writer(yaml_path) :
  with open(yaml_path, 'w') as f:
    f.write(data_yaml)

  print(f"YAML file saved to {yaml_path}")


input_zip_path_train = '/content/Train_mitotic.zip'
output_dir_train = '/content/Train_mitotic'
extract_zip_file(input_zip_path_train, output_dir_train)

"""
input_zip_path_tester= '/content/tester.zip'
output_dir_tester = '/content/tester'
extract_zip_file(input_zip_path_tester, output_dir_tester)
"""

root = r'/content/Train_mitotic/Train_mitotic'
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
print(standard_dict_mitotic)
print(standard_dict_non_mitotic)


output_img_dir = '/content/yolov8/images'
output_lbl_dir = '/content/yolov8/labels'
plot_bounding_boxes(standard_dict_mitotic, root, output_img_dir, output_lbl_dir)


split_dataset_with_augmentation(
    img_dir='/content/yolov8/images',
    lbl_dir='/content/yolov8/labels',
    output_train_img_dir='/content/yolov8/train/images',
    output_train_lbl_dir='/content/yolov8/train/labels',
    output_val_img_dir='/content/yolov8/val/images',
    output_val_lbl_dir = '/content/yolov8/val/labels' ,
    output_aug_img_dir="/content/yolov8/augmentations_images",
    output_aug_lbl_dir="/content/yolov8/aug_labels",
    split_ratio=0.3,
    augment=True
)

data_yaml = """
path: /content/yolov8  # Root directory
train: train/images  # Path to training images
val: val/images  # Path to validation images

nc: 1  # Number of classes
names: ['mitotic']  # Class names
"""


yaml_path = '/content/yolov8/data.yaml'
writer(yaml_path)
train_img_dir = '/content/yolov8/train/images'
train_lbl_dir = '/content/yolov8/train/labels'
val_img_dir = '/content/yolov8/val/images'
val_lbl_dir = '/content/yolov8/val/labels'



import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook



training_config = {
    'data': yaml_path,
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'Adam',
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'val': True
}

# Train the model
results = model.train(**training_config)

# Check available attributes in results
print("Results Attributes:")
for attr in dir(results):
    print(attr)


losses = []
precision = []
recall = []
map50 = []

# Extract metrics after training
if hasattr(results, 'metrics'):
    for epoch in range(training_config['epochs']):
        # Assuming results.metrics contains a list of metrics for each epoch
        losses.append(results.loss[epoch] if epoch < len(results.loss) else 0)
        precision.append(results.metrics['precision(B)'][epoch] if epoch < len(results.metrics['precision(B)']) else 0)
        recall.append(results.metrics['recall(B)'][epoch] if epoch < len(results.metrics['recall(B)']) else 0)
        map50.append(results.metrics['mAP50(B)'][epoch] if epoch < len(results.metrics['mAP50(B)']) else 0)
        print("This is the acitvation functions")
        if epoch % 5 == 0 and layer_name in activations:
            activation_map = activations[layer_name]
            plt.figure(figsize=(8, 6))
            plt.imshow(activation_map[0][0].cpu(), cmap='viridis')
            plt.colorbar()
            plt.title(f'Activation Map of {layer_name} at Epoch {epoch}')
            plt.show()

elif hasattr(results, 'results_dict'):
    for epoch in range(training_config['epochs']):
        losses.append(results.results_dict.get('loss', [0])[epoch] if epoch < len(results.results_dict.get('loss', [])) else 0)
        precision.append(results.results_dict.get('metrics/precision(B)', [0])[epoch] if epoch < len(results.results_dict.get('metrics/precision(B)', [])) else 0)
        recall.append(results.results_dict.get('metrics/recall(B)', [0])[epoch] if epoch < len(results.results_dict.get('metrics/recall(B)', [])) else 0)
        map50.append(results.results_dict.get('metrics/mAP50(B)', [0])[epoch] if epoch < len(results.results_dict.get('metrics/mAP50(B)', [])) else 0)

epochs = list(range(1, training_config['epochs'] + 1))

# After training, print the metrics
print("Losses:", losses)
print("Precision:", precision)
print("Recall:", recall)
print("mAP50:", map50)

# Plotting the metrics
plt.figure(figsize=(12, 8))

# Plot Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, losses, label='Loss', color='blue')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Plot Precision
plt.subplot(2, 2, 2)
plt.plot(epochs, precision, label='Precision', color='orange')
plt.title('Precision per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.grid()

# Plot Recall
plt.subplot(2, 2, 3)
plt.plot(epochs, recall, label='Recall', color='green')
plt.title('Recall per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.grid()

# Plot mAP
plt.subplot(2, 2, 4)
plt.plot(epochs, map50, label='mAP50', color='red')
plt.title('mAP50 per Epoch')
plt.xlabel('Epochs')
plt.ylabel('mAP50')
plt.grid()

plt.tight_layout()
plt.show()
