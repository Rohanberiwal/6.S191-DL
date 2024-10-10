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


dataset_list = []
from torch.utils.data import DataLoader, random_split
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch

"""
class CustomFasterRCNNDataset(Dataset):
    def __init__(self, updated_dict, transforms=None):
        self.annotations = updated_dict
        self.image_paths = list(self.annotations.keys())
        self.transforms = transforms

        # Store original dataset size
        self.original_size = len(self.image_paths)

        # Generate augmented samples
        self.augmented_data = self.generate_augmented_samples()

    def generate_augmented_samples(self):
        augmented_data = []
        for image_path in self.image_paths:
            original_boxes = self.annotations[image_path]  # List of bounding boxes for the image
            for _ in range(3):  # Create 3 augmented versions per image
                augmented_image, new_boxes = self.create_augmented_image(image_path, original_boxes)
                augmented_data.append((augmented_image, new_boxes))
        return augmented_data

    def create_augmented_image(self, image_path, boxes):
        # Load the original image
        image = Image.open(image_path).convert("RGB")

        # Apply random transformations and update bounding boxes
        if self.transforms:
            image, new_boxes = self.apply_random_transforms(image, boxes)
        else:
            new_boxes = torch.tensor(boxes, dtype=torch.float32)

        return image, new_boxes

    def apply_random_transforms(self, image, boxes):
        # Apply transformations and update bounding boxes
        if self.transforms:
            image = self.transforms(image)

        # Example transformation: horizontal flip
        if random.random() > 0.5:  # 50% chance to flip
            image = transforms.functional.hflip(image)
            # Update bounding boxes
            w = image.size[0]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # Flip x-coordinates

        # Additional transformations can be added here...

        return image, boxes

    def __len__(self):
        # Total dataset size is original + augmented
        return self.original_size + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < self.original_size:
            # Access original images
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            boxes = self.annotations[image_path]  # Get bounding boxes

            # Convert to tensor
            boxes = torch.tensor(boxes, dtype=torch.float32)

            # Assuming all labels are positive
            labels = torch.tensor([1] * len(boxes), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}

            if self.transforms:
                image = self.transforms(image)

            return image, target
        else:
            # Access augmented images
            aug_idx = idx - self.original_size
            augmented_image, boxes = self.augmented_data[aug_idx]
            # Assuming all augmented labels are positive
            labels = torch.tensor([1] * len(boxes), dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
            return augmented_image, target

# Example data transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((224, 224)),  # Resize images to a common size
])
"""

class CustomFasterRCNNDataset(Dataset):
    def __init__(self, updated_dict, transforms=None):
        self.annotations = updated_dict
        self.image_paths = list(self.annotations.keys())
        self.transforms = transforms

    def __len__(self):
        # Return the length of the dataset (original images only)
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Access original images
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        boxes = self.annotations[image_path]  # Get bounding boxes

        # Convert bounding boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Assuming all labels are positive
        labels = torch.tensor([1] * len(boxes), dtype=torch.int64)

        # Prepare the target dictionary
        target = {"boxes": boxes, "labels": labels}

        # Apply any transforms (if specified)
        if self.transforms:
            image = self.transforms(image)

        return image, target

# Example data transforms without augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
])



def custom_collate_fn(batch):
    images, targets = zip(*batch)  # Unzip the batch
    images = [ToTensor()(image) for image in images]  # Convert PIL images to tensors
    return list(images), list(targets)  # Return as lists

def create_data_loaders(dataset, train_ratio=0.7, batch_size=8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, val_loader

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

print(updated_dict)
process_histopathology_image(updated_dict)
dataset = CustomFasterRCNNDataset(updated_dict, transforms=data_transforms)
print("Len of the data set is ", len(dataset) )


from torchvision.transforms import ToTensor

train_loader, val_loader = create_data_loaders(dataset, train_ratio=0.7, batch_size=1)

print("Training Data Loader Size:", len(train_loader))
print("Validation Data Loader Size:", len(val_loader))

def extract_bounding_boxes(data_dict):
    widths = []
    heights = []
    labels = []
    for img_path, bboxes in data_dict.items():
        for bbox in bboxes:
            x, y, width, height = bbox
            widths.append(width)
            heights.append(height)
            label = 1 if 'mitotic' in img_path else 0
            labels.append(label)
    return np.array(widths), np.array(heights), np.array(labels)


widths, heights, labels = extract_bounding_boxes(updated_dict)
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.histplot(widths, bins=30, kde=True, color='blue')
plt.title('Histogram of Bounding Box Widths')
plt.xlabel('Width')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(heights, bins=30, kde=True, color='orange')
plt.title('Histogram of Bounding Box Heights')
plt.xlabel('Height')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.scatter(widths, heights, c=labels, cmap='coolwarm', alpha=0.6)
plt.title('Scatter Plot of Bounding Box Widths vs Heights')
plt.xlabel('Width')
plt.ylabel('Height')
plt.colorbar(label='Label (1: Mitotic, 0: Non-mitotic)')
plt.axhline(y=np.mean(heights), color='r', linestyle='--', label='Mean Height')
plt.axvline(x=np.mean(widths), color='g', linestyle='--', label='Mean Width')
plt.legend()

plt.tight_layout()
plt.show()




import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.optim as optim
import torchvision.models.detection as detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)
num_epochs = 10  # Adjust as needed

import torch

def print_loader_contents(data_loader):
    # Iterate through the DataLoader
    for batch_idx, (images, targets) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        
        for i in range(len(images)):
            print(f"Image {i + 1}:")
            # Print image tensor shape
            if isinstance(images[i], torch.Tensor):
                print(f"Image Tensor Shape: {images[i].shape}")
            else:
                print("Image is not a tensor.")
            
            # Print target contents
            target = targets[i]  # Access the target for the current image
            if isinstance(target, dict):
                for key, value in target.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: Tensor Shape: {value.shape}")
                    else:
                        print(f"{key}: {value}")  # Print whatever is inside the target
            else:
                print("Target is not a dictionary.")
            
            print("-" * 30)  
print_loader_contents(train_loader)


import torch
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import torch

def convert_boxes_format(boxes):
    """Convert boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    new_boxes = []
    for box in boxes:
        x, y, width, height = box
        x_min = x
        y_min = y
        x_max = x + width
        y_max = y + height
        new_boxes.append([x_min, y_min, x_max, y_max])
    return torch.tensor(new_boxes, dtype=torch.float32)

def validate_boxes(boxes):
    """Check if all boxes have positive width and height."""
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        assert x_max > x_min and y_max > y_min, f"Invalid box {box}"

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    for images, targets in data_loader:
        images = [image.to(device) for image in images]

        # Convert and validate bounding boxes
        for target in targets:
            boxes = target['boxes']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes = convert_boxes_format(boxes)
            target['boxes'] = boxes  # Update the target with new box format
            
            # Validate bounding boxes
            validate_boxes(boxes)

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return losses.item()

# Example of training
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")
