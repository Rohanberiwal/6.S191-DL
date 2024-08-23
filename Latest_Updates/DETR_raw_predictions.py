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
                
    def reset(self):
        self.meters = {}
        
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


import torch
import torch.nn as nn
import torchvision.ops as ops
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DETRLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def forward(self, pred_logits, pred_boxes, targets):
        # Compute losses for a single image and target
        loss_dict = {}
        if 'labels' in self.losses:
            loss_dict['labels'] = self.compute_classification_loss(pred_logits, targets)
        if 'boxes' in self.losses:
            loss_dict['boxes'] = self.compute_bbox_loss(pred_boxes, targets)
        if 'cardinality' in self.losses:
            loss_dict['cardinality'] = self.compute_cardinality_loss(pred_boxes, targets)
        
        # Apply weights to losses
        total_loss = 0
        for loss in self.losses:
            total_loss += self.weight_dict[loss] * loss_dict[loss]
        
        return total_loss

    def compute_classification_loss(self, pred_logits, targets):
        target_labels = targets['labels']
        print("Labels")
        print(target_labels)
        print("Logits")
        print(pred_logits)
        print("Targets")
        print(targets)
        if pred_logits.shape[0] != target_labels.shape[0]:
            raise ValueError(f"Batch size mismatch: pred_logits batch size {pred_logits.shape[0]} vs target_labels batch size {target_labels.shape[0]}")
        
        num_classes = pred_logits.shape[-1]
        eos_coef = self.eos_coef
        loss = F.cross_entropy(pred_logits, target_labels, reduction='none')
        loss = loss * (target_labels != num_classes - 1).float()
        loss = loss.sum() / (target_labels != num_classes - 1).float().sum()
        return loss

    def compute_bbox_loss(self, pred_boxes, targets):
        target_boxes = targets['boxes']
        loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
        return loss.mean()

    def compute_cardinality_loss(self, pred_boxes, targets):
        pred_cardinality = torch.tensor([len(targets['boxes'])], dtype=torch.float32)
        target_cardinality = torch.tensor([len(targets['boxes'])], dtype=torch.float32)
        loss = F.l1_loss(pred_cardinality, target_cardinality, reduction='none')
        return loss.mean()

from scipy.optimize import linear_sum_assignment
class HungarianMatcher:
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, outputs, targets):
        # Extract information
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        target_labels = targets['labels']
        target_boxes = targets['boxes']
        
        # Compute costs
        cost_class = self.compute_class_cost(pred_logits, target_labels)
        cost_bbox = self.compute_bbox_cost(pred_boxes, target_boxes)
        cost_giou = self.compute_giou_cost(pred_boxes, target_boxes)

        # Total cost
        cost_matrix = cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        return indices

    def compute_class_cost(self, pred_logits, target_labels):
        pred_logits = pred_logits.softmax(-1)
        cost_class = -pred_logits[:, :-1].log() * target_labels.unsqueeze(-1)
        return cost_class.sum(dim=-1)
    
    def compute_bbox_cost(self, pred_boxes, target_boxes):
        # L1 distance between predicted and target boxes
        pred_boxes = pred_boxes.view(-1, 4)
        target_boxes = target_boxes.view(-1, 4)
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        return cost_bbox
    
    def compute_giou_cost(self, pred_boxes, target_boxes):
        # Compute GIoU cost
        pred_boxes = pred_boxes.view(-1, 4)
        target_boxes = target_boxes.view(-1, 4)
        giou = ops.box_iou(pred_boxes, target_boxes)
        cost_giou = 1 - giou
        return cost_giou

matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

weight_dict = {
    'labels': 1.0,
    'boxes': 5.0,
    'cardinality': 0.1
}
eos_coef = 0.1
losses = ['labels', 'boxes', 'cardinality']
detr_loss = DETRLoss(num_classes, matcher, weight_dict, eos_coef, losses)

metric_logger = MetricLogger(delimiter="  ")
num_epochs = 10

def dict_loss(pred_logits, pred_boxes, targets):
    #assert pred_logits.shape[0] == len(targets['labels']), "Batch size mismatch between predictions and targets."
    target_logits = targets['labels']
    target_boxes = targets['boxes']
    
    classification_loss = F.cross_entropy(pred_logits, target_logits)
    bbox_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
    
    return {'loss_cls': classification_loss, 'loss_bbox': bbox_loss}


def print_loss_dict(loss_dict):
    print("Loss Dictionary Contents:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item()}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")



def print_model_outputs(outputs):
    print("Outputs keys:", outputs.keys())
    print("Predicted logits shape:", outputs['pred_logits'].shape)
    print("Predicted boxes shape:", outputs['pred_boxes'].shape)
    
    # Print the contents of 'pred_logits'
    print("\nPredicted logits:")
    if isinstance(outputs['pred_logits'], torch.Tensor):
        print(outputs['pred_logits'].cpu().detach().numpy()) 
    else:
        print(outputs['pred_logits'])
    
    # Print the contents of 'pred_boxes'
    print("\nPredicted boxes:")
    if isinstance(outputs['pred_boxes'], torch.Tensor):
        print(outputs['pred_boxes'].cpu().detach().numpy())  # Convert to numpy array for easier readability
    else:
        print(outputs['pred_boxes'])
    print("\nFull outputs dictionary:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.cpu().detach().numpy()}")
        else:
            print(f"{key}: {value}")
import torch


import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    intersection_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    gt_matched = [False] * len(gt_boxes)

    for pred_box in pred_boxes:
        matched = False
        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched = True
                gt_matched[i] = True
                break
        
        if matched:
            tp += 1
        else:
            fp += 1
    
    fn = gt_boxes.count(False)  # Count False in the list of matched ground truth boxes

    return tp, fp, fn


def plotter(image_tensor, pred_boxes, gt_boxes):
    to_pil = transforms.ToPILImage()
    img = to_pil(image_tensor.cpu())
    
    # Convert pred_boxes to NumPy array if necessary
    pred_boxes = pred_boxes.detach().cpu().numpy()

    draw = ImageDraw.Draw(img)
    outline_color = "#00F0F0"

    # List to store cropped patches
    cropped_patches = []

    # Debugging output
    print(f"Image size: {img.size}")
    print(f"Number of predicted boxes: {len(pred_boxes)}")

    # Plot each bounding box
    for box in pred_boxes:
        x, y, width, height = box
        x_min = int(x * img.width)  # Scale box coordinates to image size
        y_min = int(y * img.height) # Scale box coordinates to image size
        x_max = int((x + width) * img.width)  # Scale box coordinates to image size
        y_max = int((y + height) * img.height) # Scale box coordinates to image size

        # Ensure coordinates are valid and within image bounds
        x_min, x_max = max(0, min(x_min, x_max)), min(img.width, max(x_min, x_max))
        y_min, y_max = max(0, min(y_min, y_max)), min(img.height, max(y_min, y_max))

        print(f"Drawing box: (x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max})")

        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=2)

        # Crop the patch and append to list
        cropped_patch = img.crop((x_min, y_min, x_max, y_max))
        if cropped_patch.size != (0, 0):
            cropped_patches.append(cropped_patch)
        else:
            print(f"Skipped empty cropped patch at: (x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max})")

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image with Bounding Boxes')
    plt.show()

    if cropped_patches:
        for i, patch in enumerate(cropped_patches):
            plt.figure(figsize=(4, 4))
            plt.imshow(patch)
            plt.title(f'Patch {i+1}')
            plt.axis('off')
            plt.show()
    else:
        print("No cropped patches to display.")

    return cropped_patches



for epoch in range(num_epochs):
    model.train()
    metric_logger.reset()
    
    for images, targets in train_loader:
        for img, target in zip(images, targets):
            img = img.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            print("Image is this format ")
            
            print(img)
            outputs = model([img])
            print_model_outputs(outputs)
            pred_boxes = outputs['pred_boxes'][0]  
            print(pred_boxes)
            
            gt_boxes = target['boxes'].cpu().numpy() 
            plotter(img, pred_boxes, gt_boxes)
            try:
                loss_dict = dict_loss(outputs['pred_logits'], outputs['pred_boxes'], target)
                print()
                print_loss_dict(loss_dict)
                print()

            except Exception as e:
                pass
    
print("Training complete")
print("end")
