import torch
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
print(model)
print(model.state_dict().keys())

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
from PIL import Image, ImageDraw
import random 
import torchvision
import torch
import torchvision
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function to extract bounding boxes from JSON
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
import torchvision.transforms as T

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T

def plot_image(image_dict, title="Images"):
    """Helper function to plot images and their bounding boxes from a dictionary."""
    for image_path, boxes in image_dict.items():
        try:
            print(f"Trying to open image: {image_path}")
            # Open the image
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            for box in boxes:
                x, y, width, height = box
                # Draw bounding box
                draw.rectangle([x, y, x + width, y + height], outline="black", width=3)
            
            # Show the image with bounding boxes
            img.show(title)
        
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue

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
        
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_non_mitotic(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_mitotic, root)
modify_dict_inplace(standard_dict_non_mitotic, root)
print(standard_dict_mitotic)
print(standard_dict_non_mitotic)
#plot_image(standard_dict_mitotic)
#plot_image(standard_dict_non_mitotic)

def convert_bbox_format(data):
    for file_path, bboxes in data.items():
        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            x_center = x_min + (width / 2)
            y_center = y_min + (height / 2)
            bbox[:] = [x_center, y_center, width, height]

convert_bbox_format(standard_dict_mitotic)
print(standard_dict_mitotic)
print('end')

def split_data(input_dict, train_size=0.7):
    image_paths = list(input_dict.keys())
    bounding_boxes = list(input_dict.values())

    train_paths, val_paths, train_boxes, val_boxes = train_test_split(
        image_paths, bounding_boxes, train_size=train_size, random_state=42
    )

    train_dict = dict(zip(train_paths, train_boxes))
    val_dict = dict(zip(val_paths, val_boxes))

    return train_dict, val_dict


import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_path = list(self.data_dict.keys())[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = self.data_dict[img_path]
        
        if self.transform:
            img = self.transform(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.ones((len(boxes,)), dtype=torch.int64)}
        
        return img, target
    

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])


convert_bbox_format(standard_dict_mitotic)
print(standard_dict_mitotic)
print('end')

train_data_dict ,vald_data_dict = split_data(standard_dict_mitotic, train_size=0.7)
train_dataset = CustomDataset(train_data_dict, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

val_dataset = CustomDataset(vald_data_dict, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def print_dataloader_contents(dataloader, num_batches=1):

    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}:")
        print(f"Images: {images}")
        print(f"Targets: {targets}")
        print("-" * 40)

print_dataloader_contents(train_loader, num_batches=2)  
print_dataloader_contents(val_loader, num_batches=2)
print("Train code ")
num_classes = 2 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

import time
import torch

class MetricLogger:
    def __init__(self, delimiter="  "):
        self.delimiter = delimiter
        self.meters = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def log(self, data_loader, header):
        start_time = time.time()
        for batch_idx, (images, targets) in enumerate(data_loader):
            yield images, targets
            if batch_idx % 10 == 0:
                metrics = [f"{k}: {v.avg:.4f}" for k, v in self.meters.items()]
                print(f"{header} {self.delimiter.join(metrics)} Time: {time.time() - start_time:.2f}")
                start_time = time.time()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def process_batch(images, targets):
    # Check if there are at least 2 batches
    if len(images) < 2 or len(targets) < 2:
        raise ValueError("Less than 2 batches provided")

    # Process only the first 2 batches
    for img, target in zip(images[:2], targets[:2]):
        if isinstance(img, torch.Tensor):
            img = img
        else:
            raise ValueError("Image is not a torch.Tensor")

        if 'boxes' in target and 'labels' in target:
            boxes = target['boxes']
            labels = target['labels']
            
            if isinstance(boxes, torch.Tensor) and isinstance(labels, torch.Tensor):
                assert boxes.dim() == 2, "Bounding boxes should be a 2D tensor"
                assert labels.dim() == 1, "Labels should be a 1D tensor"
                num_boxes = boxes.size(0)
                assert num_boxes == labels.size(0), "Number of bounding boxes and labels must match"
                print(f"Image size: {img.size()}")
                print(f"Number of targets: {num_boxes}")
                print("Bounding boxes:", boxes)
                print("Labels:", labels)

            else:
                raise ValueError("Bounding boxes or labels are not torch.Tensor")

        else:
            raise KeyError("Target missing 'boxes' or 'labels' key")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4)


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    for images, targets in data_loader:
        # Move images and targets to the correct device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print("Images ")
        
        print(images)
        print("targets")
        print(targets)
        process_batch(images, targets)




num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer, device, epoch)

print("Training complete.")
