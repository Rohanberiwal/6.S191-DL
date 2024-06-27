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
                labels.append(1)  # Mitotic label

        if img_path in self.non_mitotic_dict:
            for box in self.non_mitotic_dict[img_path]:
                boxes.append(box)
                labels.append(0)  # Non-mitotic label

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Filter out invalid boxes
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

# Function to get transformations
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

# Function to print mitotic information from JSON
def print_mitotic(json_mitotic):
    print("This is the mitotic printer function")
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
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    return standard_dict_mitotic

# Function to print non-mitotic information from JSON
def print_filename_bbox(json_file):
    print("This is the printer filename function for non-mitotic")
    standard_dict_non_mitotic = {}
    universal_list = []
    with open(json_file, 'r') as f:
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
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    return standard_dict_non_mitotic

# Function to extract filenames from JSON
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

# Function to modify dictionary keys inplace
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

# Define root and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Extract filenames and modify dicts
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)

modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)

# Create dataset and data loader
dataset = CustomDataset(standard_dict_mitotic, standard_dict_non_mitotic, transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier head
num_classes = 2  # 1 class (object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 20
losses_per_epoch = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    epoch_loss = running_loss / len(data_loader)
    losses_per_epoch.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    lr_scheduler.step()

# Plotting the loss
plt.figure(figsize=(20, 5))
plt.plot(range(1, num_epochs + 1), losses_per_epoch, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()

# Testing phase
model.eval()

# Define transformations
transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800))  # Resize image to match training input size
])

# Test image paths
test_image_paths = [
   r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
]

for test_image_path in test_image_paths:
    # Load and preprocess the test image
    image = Image.open(test_image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
        print(predictions)
        
    # Print predictions
    print(f"Predictions for image: {test_image_path}")
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if label == 1:  
            print(f"Prediction: Class {'Mitotic'}, Score {score:.2f}, Bounding Box: {box}")

    # Visualize predictions (optional)
    def visualize_predictions(image, predictions):
        draw = ImageDraw.Draw(image)
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if label == 1:  # Mitotic label
                xmin, ymin, xmax, ymax = box
                color = 'green'
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                draw.text((xmin, ymin), f'Mitotic {score:.2f}', fill=color)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    original_image = T.ToPILImage()(image_tensor.squeeze(0).cpu())
    visualize_predictions(original_image, predictions)

print("Training and testing complete!")



# Testing phase
model.eval()

# Define transformations
transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800))  # Resize image to match training input size
])

# Test image paths
test_image_paths = [
   r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg" , r"C:\Users\rohan\OneDrive\Desktop\input.jpg"
]

for test_image_path in test_image_paths:
    # Load and preprocess the test image
    image = Image.open(test_image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) 
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    print(f"Predictions for image: {test_image_path}")
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if label == 1:  # Mitotic label
            print(f"Prediction: Class {'Mitotic'}, Score {score:.2f}, Bounding Box: {box}")

    # Visualize mitotic predictions (optional)
    def visualize_mitotic_predictions(image, predictions):
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if label == 1:  # Mitotic label
                xmin, ymin, xmax, ymax = box
                color = 'green'
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                draw.text((xmin, ymin), f'Mitotic {score:.2f}', fill=color)

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    original_image = T.ToPILImage()(image_tensor.squeeze(0).cpu())
    visualize_mitotic_predictions(original_image, predictions)

print("Testing complete!")
