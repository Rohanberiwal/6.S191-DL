import os
import json
import zipfile
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

from ultralytics import YOLO
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


import random
from PIL import ImageEnhance
import os
import imgaug.augmenters as iaa
import numpy as np


import matplotlib.pyplot as plt
from PIL import Image

def collect_mitotic_data(image_dict, output_dir='Images'):
    mitotic_data = []
    os.makedirs(output_dir, exist_ok=True)
    for image_path, boxes in image_dict.items():
        original_image = Image.open(image_path)

        for i, box in enumerate(boxes):
            x, y, width, height = box
            cropped_img = original_image.crop((x, y, x + width, y + height))

            # Save the cropped image with a unique name
            cropped_image_path = os.path.join(output_dir, f'crop_{os.path.basename(image_path).split(".")[0]}_{i}.png')
            cropped_img.save(cropped_image_path)

            mitotic_data.append({
                'original_image_path': image_path,  # Save original image path
                'cropped_image_path': cropped_image_path,  # Save cropped image path
                'bounding_box': box,
                'label': 1
            })

    return mitotic_data

def adjust_bounding_box(bbox, image_size, augmented_image_size):
    scale_x = augmented_image_size[0] / image_size[0]
    scale_y = augmented_image_size[1] / image_size[1]

    x_min = int(bbox[0] * scale_x)
    y_min = int(bbox[1] * scale_y)
    width = int(bbox[2] * scale_x)
    height = int(bbox[3] * scale_y)

    return [x_min, y_min, width, height]

def augment_image(image, bbox):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-25, 25)),
        iaa.Multiply((0.8, 1.2)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
    ])

    image_np = np.array(image)
    image_aug = seq(image=image_np)

    return Image.fromarray(image_aug)

def augment_mitotic_data(mitotic_data, num_augmentations=10000):
    augmented_images_dir = '/content/Images/'
    os.makedirs(augmented_images_dir, exist_ok=True)

    augmented_data = []

    for i in range(num_augmentations):
        sample = random.choice(mitotic_data)
        cropped_image_path = sample['cropped_image_path']  # Use the cropped image path
        bbox = sample['bounding_box']
        label = sample['label']
        image = Image.open(cropped_image_path)
        original_size = image.size

        augmented_image = augment_image(image, bbox)
        augmented_image_size = augmented_image.size

        adjusted_bbox = adjust_bounding_box(bbox, original_size, augmented_image_size)

        augmented_image_path = os.path.join(augmented_images_dir, f'aug_{i}.png')
        augmented_image.save(augmented_image_path)

        augmented_data.append({
            'image_path': augmented_image_path,
            'bounding_box': adjusted_bbox,
            'label': label
        })

    return augmented_data

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


def plot_scatter(augmented_data):
    bounding_boxes = np.array([data['bounding_box'] for data in augmented_data])
    plt.figure(figsize=(10, 6))
    plt.scatter(bounding_boxes[:, 0], bounding_boxes[:, 1], label='Top-left Corner', alpha=0.5)
    plt.scatter(bounding_boxes[:, 2] + bounding_boxes[:, 0], bounding_boxes[:, 3] + bounding_boxes[:, 1],
                label='Bottom-right Corner', alpha=0.5, color='r')

    plt.title('Scatter Plot of Bounding Box Corners')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.legend()
    plt.grid()
    plt.show()


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
mitotis_data =  collect_mitotic_data(standard_dict_mitotic, output_dir='Images')


plot_bounding_boxes(mitotis_data, "Bounding Boxes Before Normalization")


import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, mitotic_data, transform=None):
        self.mitotic_data = mitotic_data
        self.transform = transform

    def __len__(self):
        return len(self.mitotic_data)

    def __getitem__(self, idx):
        entry = self.mitotic_data[idx]
        image = Image.open(entry['cropped_image_path']).convert("RGB")
        bbox = list(map(float, entry['bounding_box']))
        label = entry['label']

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': torch.tensor([bbox], dtype=torch.float32),
            'labels': torch.tensor([label], dtype=torch.int64)
        }

        return image, target


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

def create_dataset(mitotic_data, batch_size=16, train_ratio=0.8):
    dataset = CustomDataset(mitotic_data , transform=transform)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


train_loader , val_loader = create_dataset(mitotis_data , batch_size=16, train_ratio=0.8)
print(train_loader)
print(val_loader)

import os
from PIL import Image

import os
import random
from PIL import Image

import os
from PIL import Image

import os
import random
from PIL import Image

def save_dataset_with_labels(mitotic_data, output_dir='output_dataset', train_ratio=0.8):
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle the data
    random.shuffle(mitotic_data)

    # Split the data into training and validation sets
    split_index = int(len(mitotic_data) * train_ratio)
    train_data = mitotic_data[:split_index]
    val_data = mitotic_data[split_index:]

    # Create directories for training and validation
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Save training data
    save_data(train_data, train_dir)

    save_data(val_data, val_dir)

def save_data(data, subset_dir):
    images_dir = os.path.join(subset_dir, 'images')
    labels_dir = os.path.join(subset_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for entry in data:
        cropped_image_path = entry['cropped_image_path']
        bbox = entry['bounding_box']
        label = entry['label']
        image = Image.open(cropped_image_path).convert("RGB")
        image_name = os.path.basename(cropped_image_path).replace('.png', '.jpg')
        image_output_path = os.path.join(images_dir, image_name)
        image.save(image_output_path)

        class_id = 1  # Set class ID to 1 for the mitotic class
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        img_width, img_height = image.size
        x_center_normalized = x_center / img_width
        y_center_normalized = y_center / img_height
        width_normalized = width / img_width
        height_normalized = height / img_height

        if (x_center_normalized < 0 or x_center_normalized > 1 or
            y_center_normalized < 0 or y_center_normalized > 1 or
            width_normalized < 0 or width_normalized > 1 or
            height_normalized < 0 or height_normalized > 1):
            print(f"Warning: Invalid bounding box for image {image_name}. Skipping this entry.")
            continue

        label_output_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
        with open(label_output_path, 'w') as label_file:
            label_file.write(f"{class_id} {x_center_normalized:.6f} {y_center_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}\n")



save_dataset_with_labels(mitotis_data, output_dir='/content/store_datasets')
print("Dataset saved successfully.")
print("end")

import yaml
from ultralytics import YOLO
data_yaml = {
    'train': '/content/store_datasets/train',
    'val': '/content/store_datasets/val',
    'nc': 2,
    'names': ['mitotic']
}
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
