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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
validation_loss = []
class CustomDataset(Dataset):
    def __init__(self, mitotic_dict, non_mitotic_dict, transforms=None):
        self.mitotic_dict = mitotic_dict
        self.non_mitotic_dict = non_mitotic_dict
        self.transforms = transforms
        self.image_files = list(set(mitotic_dict.keys()).union(set(non_mitotic_dict.keys())))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []

        if img_path in self.mitotic_dict:
            for box in self.mitotic_dict[img_path]:
                boxes.append(box)
                labels.append(1) 

        if img_path in self.non_mitotic_dict:
            for box in self.non_mitotic_dict[img_path]:
                boxes.append(box)
                labels.append(0)  

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_indices]
        labels = labels[valid_indices]

        if self.transforms:
            img = self.transforms(img)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)

        return img, target


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

def plot_images(images, targets, title="Images"):
    """Helper function to plot images and their bounding boxes."""
    fig, ax = plt.subplots(1, len(images), figsize=(12, 6))
    if len(images) == 1:
        ax = [ax]
    for i, (image, target) in enumerate(zip(images, targets)):
        ax[i].imshow(T.ToPILImage()(image).convert("RGB"))
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 fill=False, edgecolor='red', linewidth=3)
            ax[i].add_patch(rect)
        ax[i].set_title(f"Image {i+1}")
        ax[i].axis('off')
    plt.show()


def print_dataloader(dataloader, num_batches=1):
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i+1}:")
        print(f"Number of images in batch: {len(images)}")
        print("Sample images and targets:")
        plot_images(images, targets)

def print_loss_dict(loss_dict):
    if isinstance(loss_dict, list):
        for i, loss_entry in enumerate(loss_dict):
            print(f"Entry {i + 1}:")
            if isinstance(loss_entry, dict):
                for key, value in loss_entry.items():
                    print(f"  {key}:")
                    if isinstance(value, torch.Tensor):
                        print(f"    Tensor shape: {value.shape}")
                        print(f"    Tensor values: {value.tolist()}")
                        validation_loss.append(value.tolist())
                        
                    else:
                        print(f"    Value: {value}")
            else:
                print(f"  Unexpected format: {loss_entry}")
            print()
    else:
        print(f"Unexpected type for loss_dict: {type(loss_dict)}")

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
print(standard_dict_mitotic.keys())
print(standard_dict_non_mitotic.keys())
dataset_keys = list(set(standard_dict_mitotic.keys()).union(set(standard_dict_non_mitotic.keys())))
print(dataset_keys)

train_size = int(0.7 * len(dataset_keys))
val_size =  int(0.3*len(dataset_keys))
common_keys = set(standard_dict_mitotic.keys()).intersection(set(standard_dict_non_mitotic.keys()))
train_keys = list(common_keys)[:train_size]
val_keys = list(common_keys)[train_size:]


train_dataset = CustomDataset(
    {key: standard_dict_mitotic[key] for key in train_keys},
    {key: standard_dict_non_mitotic[key] for key in train_keys},
    transforms=get_transform(train=True)
)

val_dataset = CustomDataset(
    {key: standard_dict_mitotic[key] for key in val_keys},
    {key: standard_dict_non_mitotic[key] for key in val_keys},
    transforms=get_transform(train=False)
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
print("Training Data Loader:")
print_dataloader(train_loader)
print("Validation Data Loader:")
print_dataloader(val_loader)
print(len(train_loader))
print(len(val_loader))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
import numpy as np 
print("This is the train loop  ")


import matplotlib.patches as patches
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    return img

def plot_image_with_boxes(image_path, boxes, scores, labels, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    
    ax = plt.gca()
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold and label == 1:  # Filter for mitotic regions
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
            print(f"Mitotic Region Bounding Box: {box}, Score: {score}")

    plt.axis('off')
    plt.show()


num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss_accum = 0.0
    
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict[key].mean() for key in loss_dict.keys())
        total_loss.backward()
        optimizer.step()
        train_loss_accum += total_loss.item()
    
    train_loss = train_loss_accum / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    
    model.eval()
    val_loss_accum = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            print(loss_dict)
            print_loss_dict(loss_dict)

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
print("This is the test  phase of the faster rcnnn ")
test_image = preprocess_image(image_path)
test_image = test_image.unsqueeze(0).to(device)
with torch.no_grad():
    prediction = model(test_image)
    print(prediction)
    
print("end")

