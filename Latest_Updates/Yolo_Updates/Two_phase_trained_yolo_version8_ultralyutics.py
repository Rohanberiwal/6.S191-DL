import os
import json
import os
import zipfile
import random
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from PIL import Image
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
dataset_list = []

from torch.utils.data import DataLoader, random_split
import torch

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

model = ssdlite320_mobilenet_v3_large(pretrained=True)
print(model)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

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
    print("Printing mitotic information:")
    standard_dict_mitotic = {}
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
            boundary_box.append([xmin, ymin, width, height])
            universal_list.append([xmin, ymin, width, height])
        print("------------------------")
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    return standard_dict_mitotic


def print_non_mitotic(json_non_mitotic):
    print("Printing non-mitotic information:")
    standard_dict_non_mitotic = {}
    universal_list = []
    with open(json_non_mitotic, 'r') as f:
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
            boundary_box.append([xmin, ymin, width, height])
            universal_list.append([xmin, ymin, width, height])
        print("------------------------")
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
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


import matplotlib.pyplot as plt
from PIL import Image

from PIL import Image
import os

def collect_mitotic_data(image_dict, output_dir='Images'):
    mitotic_data = []
    os.makedirs(output_dir, exist_ok=True)

    for image_path, boxes in image_dict.items():
        original_image = Image.open(image_path)
        image_width, image_height = original_image.size
        for i, box in enumerate(boxes):
            x, y, width, height = box
            if width <= 0 or height <= 0 or x < 0 or y < 0 or (x + width) > image_width or (y + height) > image_height:
                print(f"Invalid bounding box for cropping: {box} (out of bounds or negative)")
                continue


            cropped_img = original_image.crop((x, y, x + width, y + height))
            cropped_image_path = os.path.join(output_dir, f'crop_{os.path.basename(image_path).split(".")[0]}_{i}.png')
            cropped_img.save(cropped_image_path)

            adjusted_box = [0, 0, width, height]
            mitotic_data.append({
                'original_image_path': image_path,
                'cropped_image_path': cropped_image_path,
                'bounding_box': adjusted_box,
                'label': 0
            })

    return mitotic_data



def plot_cropped_bounding_boxes(image_dict):
    for image_path, boxes in image_dict.items():
        img = Image.open(image_path)
        for box in boxes:
            x, y, width, height = box

            cropped_img = img.crop((x, y, x + width, y + height))

            plt.figure(figsize=(5, 5))
            plt.imshow(cropped_img)
            plt.axis('off')
            plt.title(f"Cropped: {image_path}")
            plt.show()


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

import random

def split_data(image_dict, validation_split=0.2):
    data_items = list(image_dict.items())
    random.shuffle(data_items)
    validation_size = int(len(data_items) * validation_split)
    validation_data = dict(data_items[:validation_size])
    training_data = dict(data_items[validation_size:])
    return training_data, validation_data

def extract_zip_file(input_zip_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Files extracted to {output_dir}")

def convert_bbox_xywh_to_minmax(bbox):
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]


def create_masked_image(image_path, bounding_boxes):
    image = Image.open(image_path).convert("L")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    for bbox in bounding_boxes:
        x, y, width, height = bbox
        draw.rectangle([x, y, x + width, y + height], fill=255)
    return mask


def draw_bounding_boxes(image_path, bounding_boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for bbox in bounding_boxes:
        x, y, width, height = bbox
        draw.rectangle([x, y, x + width, y + height], outline="blue", width=2)

    return image

def process_histopathology_image(updated_dict):
    """
    for image_path, bounding_boxes in updated_dict.items():
        image_with_bboxes = create_masked_image(image_path, bounding_boxes)
        #draw_bounding_boxes(image_path, bounding_boxes)
        plt.imshow(image_with_bboxes)
        plt.axis('off')
        plt.show()
    """



def print_statistics(bounding_boxes):
    mean = np.mean(bounding_boxes, axis=0)
    variance = np.var(bounding_boxes, axis=0)
    median = np.median(bounding_boxes, axis=0)

    print("Statistics:")
    print("Mean:", mean)
    print("Variance:", variance)
    print("Median:", median)

def normalize_mitotic_data(mitotic_data):
    bounding_boxes = np.array([data['bounding_box'] for data in mitotic_data])
    print_statistics(bounding_boxes)
    mean_values = bounding_boxes.mean(axis=0)
    std_values = bounding_boxes.std(axis=0)
    normalized_boxes = (bounding_boxes - mean_values) / std_values
    normalized_boxes[:, 2] = np.clip(normalized_boxes[:, 2], 0.01, None)  # Width
    normalized_boxes[:, 3] = np.clip(normalized_boxes[:, 3], 0.01, None)
    for i, data in enumerate(mitotic_data):
        data['bounding_box'] = normalized_boxes[i].tolist()
    print_statistics(normalized_boxes)
    return mitotic_data

def plot_bounding_boxes(mitotic_data, title):
    bounding_boxes = np.array([data['bounding_box'] for data in mitotic_data])
    plt.figure(figsize=(10, 6))
    plt.scatter(bounding_boxes[:, 0], bounding_boxes[:, 1], label='Top-left Corner', alpha=0.5)
    plt.scatter(bounding_boxes[:, 2] + bounding_boxes[:, 0], bounding_boxes[:, 3] + bounding_boxes[:, 1],
                label='Bottom-right Corner', alpha=0.5, color='r')

    plt.title(title)
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

def plot_hyperplane(augmented_data):
    bounding_boxes = np.array([data['bounding_box'] for data in augmented_data])
    x = bounding_boxes[:, 0]
    y = bounding_boxes[:, 1]

    a, b, c = 1, -1, 0

    x_range = np.linspace(min(x), max(x), 100)
    y_range = -(a * x_range + c) / b

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Bounding Boxes', alpha=0.6)
    plt.plot(x_range, y_range, color='red', label='Hyperplane: ax + by + c = 0')

    plt.title('Bounding Boxes and Hyperplane')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()


def convert_and_save_bounding_boxes(standard_dict, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    for img_path, bboxes in standard_dict.items():
        img = Image.open(img_path)
        img_width, img_height = img.size
        img_filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_filename)
        bbox_labels = []
        image_patches_info = []

        for i, bbox in enumerate(bboxes):
            bbox_minmax = convert_bbox_xywh_to_minmax(bbox)
            x_min, y_min, x_max, y_max = bbox_minmax
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            cropped_img_filename = f"{base_name}_bbox_{i}.png"
            cropped_img_path = os.path.join(output_img_dir, cropped_img_filename)
            cropped_img.save(cropped_img_path)

            bbox_label = f"{x_min} {y_min} {x_max} {y_max}"
            label = 1
            bbox_labels.append((bbox_label, label))

            image_patches_info.append({
                'path': cropped_img_path,
                'bbox': bbox_label,
                'label': label
            })

        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(output_txt_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for bbox_label, label in bbox_labels:
                f.write(f"{bbox_label}\n{label}\n")

        dataset_list.extend(image_patches_info)

    return standard_dict


def print_mitotic_data(mitotic_data):
    for idx, data in enumerate(mitotic_data):
        print(f"Data #{idx + 1}:")
        print(f"  Original Image Path: {data['original_image_path']}")
        print(f"  Cropped Image Path: {data['cropped_image_path']}")
        print(f"  Bounding Box: {data['bounding_box']}")
        print(f"  Label: {data['label']}")
        print("-" * 40)



import random
from math import floor

def split_data(input_dict, split_ratio=0.8):
    train_data = {}
    test_data = {}

    for file_path, bbox_list in input_dict.items():
        random.shuffle(bbox_list)
        split_index = floor(len(bbox_list) * split_ratio)
        train_data[file_path] = bbox_list[:split_index]
        test_data[file_path] = bbox_list[split_index:]

    return train_data, test_data

input_zip_path_train = '/content/Train_mitotic.zip'
output_dir_train = '/content/Train_mitotic'
extract_zip_file(input_zip_path_train, output_dir_train)


input_zip_path_tester= '/content/tester.zip'
output_dir_tester = '/content/tester'
extract_zip_file(input_zip_path_tester, output_dir_tester)

root = '/content/Train_mitotic/Train_mitotic'
mitotic_annotation_file = 'mitotic.json'
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
modify_dict_inplace(standard_dict_mitotic, root)

print("This is the input standard dict ",standard_dict_mitotic)
train_output =  '/content/Faster_RCNN/patch'
train_labeled = '/content/Faster_RCNN/patch_contents'
updated_dict = convert_and_save_bounding_boxes(standard_dict_mitotic, train_output, train_labeled)


#print(updated_dict)
mitotic_data = collect_mitotic_data(updated_dict)
print("This is the code for mitotc data")
print(mitotic_data)




import os
import random
from PIL import Image

def save_cropped_patches(mitotic_data, output_dir='cropped_patches', val_split=0.2):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

    # Split data into train and validation sets
    random.shuffle(mitotic_data)
    num_val = int(len(mitotic_data) * val_split)
    train_data = mitotic_data[:-num_val]
    val_data = mitotic_data[-num_val:]

    def save_image_and_label(data_split, cropped_image_path, label_filename, image, box):
        # Save the cropped image
        image.save(cropped_image_path)

        # Save the label file in YOLO format
        with open(label_filename, 'w') as label_file:
            x, y, width, height = box
            image_width, image_height = image.size
            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height
            width_normalized = width / image_width
            height_normalized = height / image_height
            label_file.write(f"0 {x_center} {y_center} {width_normalized} {height_normalized}\n")

    # Process the data
    all_train_images = []
    all_val_images = []

    for data in train_data:
        cropped_image_path = os.path.join(output_dir, 'train', 'images', os.path.basename(data['cropped_image_path']))
        label_filename = os.path.join(output_dir, 'train', 'labels', f"{os.path.basename(data['cropped_image_path']).split('.')[0]}.txt")
        image = Image.open(data['cropped_image_path'])
        save_image_and_label('train', cropped_image_path, label_filename, image, data['bounding_box'])
        all_train_images.append(cropped_image_path)

    for data in val_data:
        cropped_image_path = os.path.join(output_dir, 'val', 'images', os.path.basename(data['cropped_image_path']))
        label_filename = os.path.join(output_dir, 'val', 'labels', f"{os.path.basename(data['cropped_image_path']).split('.')[0]}.txt")
        image = Image.open(data['cropped_image_path'])
        save_image_and_label('val', cropped_image_path, label_filename, image, data['bounding_box'])
        all_val_images.append(cropped_image_path)

    # Save the YAML file in the correct path
    yaml_file_path = os.path.join('/content/', 'patches.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(f"train: {os.path.abspath(os.path.join(output_dir, 'train'))}\n")
        yaml_file.write(f"val: {os.path.abspath(os.path.join(output_dir, 'val'))}\n")
        yaml_file.write("nc: 1\n")  # Number of classes
        yaml_file.write("names: ['mitotic']\n")  # Class names

    print(f"Dataset organized and saved in {output_dir}. YAML file generated: {yaml_file_path}")


# Call the function to save the patches and YAML file
save_cropped_patches(mitotic_data, output_dir='cropped_patches', val_split=0.2)
from ultralytics import YOLO

# Define the path to the YAML file
yaml_path = '/content/patches.yaml'

# Define the model and training parameters
device = 'cpu'  # Use 'cuda' if you have GPU available
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

epochs = 80   # Number of training epochs
img_size = 640     # Image size (640x640)
batch_size = 16    # Batch size
learning_rate = 0.01 

# Start training
model.train(
    data=yaml_path,  # Path to YAML config file
    epochs=epochs,    # Number of epochs
    batch=batch_size, # Batch size
    imgsz=img_size,   # Image size
    device=device,    # Device to use ('cpu' or 'cuda')
    project='/content/yolov8_output',  # Path to save training output
    name='mitotic_train',  # Training run name
    exist_ok=True,    # Allow overwriting existing outputs
    save=True,        # Save model checkpoints
    save_period=1,    # Save every epoch
    verbose=True      # Display training logs
)


def convert_to_yolo_format_and_split(input_dict, output_dir='dataset_yolo', val_split=0.2, label=0):

    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

    images = list(input_dict.keys())
    random.shuffle(images)

    # Split the dataset into training and validation
    num_val = int(len(images) * val_split)
    train_images = images[:-num_val]
    val_images = images[-num_val:]

    def save_image_and_label(image_path, bbox_list, data_split):  

        original_image = Image.open(image_path)
        image_width, image_height = original_image.size
        
        # Save the image
        image_filename = os.path.join(output_dir, data_split, 'images', os.path.basename(image_path))
        original_image.save(image_filename)
        
        # Save the label file
        label_filename = os.path.join(output_dir, data_split, 'labels', f"{os.path.basename(image_path).split('.')[0]}.txt")
        with open(label_filename, 'w') as label_file:
            for bbox in bbox_list:
                x, y, width, height = bbox
                # Convert to YOLO format (normalized coordinates)
                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                width_normalized = width / image_width
                height_normalized = height / image_height
                # Always use class index 0
                label_file.write(f"0 {x_center} {y_center} {width_normalized} {height_normalized}\n")

    # Process train images
    for image_path in train_images:
        save_image_and_label(image_path, input_dict[image_path], 'train')

    # Process validation images
    for image_path in val_images:
        save_image_and_label(image_path, input_dict[image_path], 'val')

    print(f"Dataset split into train and val, YOLO format saved to {output_dir}")
convert_to_yolo_format_and_split(updated_dict, output_dir='dataset_yolo', val_split=0.2, label=1)


import yaml
import os

def create_yolo_yaml(train_dir, val_dir, yaml_path, num_classes=1, class_names=['mitotic']):
    data_dict = {
        'train': os.path.join(train_dir, 'images'),
        'val': os.path.join(val_dir, 'images'),
        'nc': num_classes,
        'names': class_names
    }

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data_dict, yaml_file, default_flow_style=False)

    print(f"YAML configuration saved at {yaml_path}")
train_dir = '/content/dataset_yolo/train'
val_dir = '/content/dataset_yolo/val'
yaml_path = '/content/dataset.yaml'

create_yolo_yaml(train_dir, val_dir, yaml_path, num_classes=1, class_names=['mitotic'])

from ultralytics import YOLO

# Path to your YAML config
yaml_path = '/content/dataset.yaml'

# Initialize YOLO model (use 'yolov8n.pt' or other versions if you want a pretrained model)
model = YOLO('yolov8n.yaml')  # Use the model config or pretrained weights

# Define training parameters
epochs = 80  # Start with fewer epochs for debugging
batch_size = 4  # Reduce batch size if you're running into memory issues
img_size = 640  # Image size (adjust if necessary)
device = 'cpu'  # Use 'cpu' or 'cuda' for GPU

# Start training
model.train(
    data=yaml_path,  # Path to YAML config file
    epochs=epochs,    # Number of epochs
    batch=batch_size, # Batch size
    imgsz=img_size,   # Image size (640x640)
    device=device,    # Device to use ('cpu' or 'cuda')
    project='/content/yolov8_output',  # Path to save training output
    name='mitotic_train',  # Training run name
    exist_ok=True,    # Allow overwriting existing outputs
    save=True,        # Save model checkpoints
    save_period=1,    # Save every epoch (useful for debugging)
    verbose=True      # Display training logs
)
