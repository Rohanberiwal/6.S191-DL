import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import os
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import zipfile
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T

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



def extract_zip_file(input_zip_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Files extracted to {output_dir}")


import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
"""
def convert_bbox_xywh_to_minmax(bbox):
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def convert_and_save_bounding_boxes(standard_dict, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    
    for img_path, bboxes in standard_dict.items():
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        img_width, img_height = img.size
        img_filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_filename)
        
        bbox_labels = []
        
        for i, bbox in enumerate(bboxes):
            bbox_minmax = convert_bbox_xywh_to_minmax(bbox)
            x_min, y_min, x_max, y_max = bbox_minmax
            
            # Draw rectangle on image
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
            
            # Crop the image patch
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            cropped_img_filename = f"{base_name}_bbox_{i}.png"
            cropped_img_path = os.path.join(output_img_dir, cropped_img_filename)
            cropped_img.save(cropped_img_path)
            
            # Prepare bounding box and label
            bbox_label = f"Bounding Box {i + 1}: {x_min} {y_min} {x_max} {y_max}"
            label = "Label: 1"
            bbox_labels.append((bbox_label, label))
        
        # Save bounding boxes and labels in a text file
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(output_txt_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for bbox_label, label in bbox_labels:
                f.write(f"{bbox_label}\n{label}\n")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return standard_dict
"""


import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


dataset_list = []

def convert_bbox_xywh_to_minmax(bbox):
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def convert_and_save_bounding_boxes(standard_dict, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    
    for img_path, bboxes in standard_dict.items():
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        img_width, img_height = img.size
        img_filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_filename)
        
        bbox_labels = []
        image_patches_info = []

        for i, bbox in enumerate(bboxes):
            bbox_minmax = convert_bbox_xywh_to_minmax(bbox)
            x_min, y_min, x_max, y_max = bbox_minmax
            
    
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

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
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        dataset_list.extend(image_patches_info)

    return standard_dict


import torch
from torchvision.io import read_image
from torchvision.transforms import functional as F
from PIL import Image

import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = item['path']
        bbox = list(map(float, item['bbox'].split()))
        label = item['label']

        image = Image.open(image_path).convert("RGB")

        bbox = torch.tensor(bbox, dtype=torch.float32).reshape(-1, 4)
        label = torch.tensor(label, dtype=torch.int64)
        
        target = {
            'boxes': bbox,
            'labels': label
        }

        if self.transform:
            image, target = self.transform(image, target)
        
        return F.to_tensor(image), target

class AugmentationTransform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    
    def __call__(self, image, target):
        image = self.transforms(image)
        return image, target


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

print(standard_dict_mitotic)
#plot_image(standard_dict_mitotic)

train_output =  '/content/Faster_RCNN/patch'
train_labeled = '/content/Faster_RCNN/patch_contents'
updated_dict = convert_and_save_bounding_boxes(standard_dict_mitotic, train_output, train_labeled)

print(updated_dict)
#plot_bounding_boxes(standard_dict_mitotic, root, output_img_dir, output_lbl_dir)

print(dataset_list)

train_list, val_list = train_test_split(dataset_list, test_size=0.3, random_state=42)

augmentation_transform = AugmentationTransform()
train_dataset = CustomDataset(annotations=train_list, transform=augmentation_transform)
val_dataset = CustomDataset(annotations=val_list, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Print detailed loss information
        print("Loss Dictionary:")
        for key, value in loss_dict.items():
            print(f"{key}: {value.item()}")
        
        total_loss += losses.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Print detailed loss information
            print("Validation Loss Dictionary:")
            for key, value in loss_dict.items():
                print(f"{key}: {value.item()}")
            
            total_loss += losses.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=10, device='cuda'):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        print("Training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        print("Validating...")
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")


train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=10, device='cuda')
print("code ends")
"""
import torch

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print("Loss Dictionary:")
        for key, value in loss_dict.items():
            print(f"{key}: {value.item()}")
        
        total_loss += losses.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            print("Validation Loss Dictionary:")
            for key, value in loss_dict.items():
                print(f"{key}: {value.item()}")
            
            total_loss += losses.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=10, device='cpu'):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        print("Training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        print("Validating...")
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

device = torch.device('cpu')

model.to(device)

train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=10, device=device)
print("code ends")
