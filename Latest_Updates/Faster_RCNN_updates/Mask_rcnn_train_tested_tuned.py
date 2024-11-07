import torch
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
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
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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

def collect_mitotic_data(image_dict, output_dir='Images'):
    mitotic_data = []
    os.makedirs(output_dir, exist_ok=True)

    for image_path, boxes in image_dict.items():
        original_image = Image.open(image_path)

        for i, box in enumerate(boxes):
            x, y, width, height = box
            cropped_img = original_image.crop((x, y, x + width, y + height))
            cropped_image_path = os.path.join(output_dir, f'crop_{os.path.basename(image_path).split(".")[0]}_{i}.png')
            cropped_img.save(cropped_image_path)

            mitotic_data.append({
                'original_image_path': image_path,
               'cropped_image_path': cropped_image_path,
                'bounding_box': box,
                'label': 1
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



def extract_zip_file(input_zip_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Files extracted to {output_dir}")


def process_histopathology_image(updated_dict):
    """
    for image_path, bounding_boxes in updated_dict.items():
        image_with_bboxes = create_masked_image(image_path, bounding_boxes)
        #draw_bounding_boxes(image_path, bounding_boxes)
        plt.imshow(image_with_bboxes)
        plt.axis('off')
        plt.show()
    """



def plot_bounding_boxes(mitotic_data, title):
    bounding_boxes = np.array([data['bbox'] for data in mitotic_data])
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
"""


import os
from PIL import Image, ImageDraw

import os
from PIL import Image, ImageDraw

def convert_bbox_xywh_to_minmax(bbox):
    """Convert bounding box from (x, y, width, height) to (x_min, y_min, x_max, y_max)."""
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def create_masked_image(image_path, bbox):
    """Create a mask image from a single bounding box."""
    image = Image.open(image_path).convert("L")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

    return mask

def draw_bounding_box(image_path, bbox):
    """Draw a single bounding box on the image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)

    return image

def convert_and_save_bounding_boxes(standard_dict, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    dataset_list = []
    for img_path, bboxes in standard_dict.items():
        img_filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_filename)

        for i, bbox in enumerate(bboxes):
            # Convert bounding box to minmax format
            bbox_minmax = convert_bbox_xywh_to_minmax(bbox)

            # Create the mask for the bounding box
            mask = create_masked_image(img_path, bbox_minmax)

            # Draw the bounding box on the image
            modified_img = draw_bounding_box(img_path, bbox_minmax)

            # Save the modified image and mask
            modified_img_filename = f"{base_name}_bbox_{i}.png"
            modified_img_path = os.path.join(output_img_dir, modified_img_filename)
            modified_img.save(modified_img_path)

            mask_filename = f"{base_name}_mask_{i}.png"
            mask_path = os.path.join(output_img_dir, mask_filename)
            mask.save(mask_path)

            bbox_label = f"{bbox_minmax[0]} {bbox_minmax[1]} {bbox_minmax[2]} {bbox_minmax[3]}"
            label = 1  # Change this according to your labeling scheme
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(output_txt_dir, txt_filename)
            with open(txt_path, 'a') as f:
                f.write(f"{bbox_label} {label}\n")

            dataset_list.append({
                'path': modified_img_path,
                'bbox': bbox_label,
                'label': label,
                'mask_path': mask_path
            })
    return dataset_list

import torch
from torch.utils.data import Dataset
from PIL import Image


from torchvision import transforms
from PIL import Image
import torch

def normalize_dataset(dataset):
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    normalized_dataset = []

    for entry in dataset:
        image = Image.open(entry['path']).convert("RGB")
        bbox = list(map(float, entry['bbox'].split()))
        label = entry['label']
        mask = Image.open(entry['mask_path']).convert("L") if entry['mask_path'] else None

        image = normalize_transform(image)
        width, height = image.shape[1], image.shape[2]
        bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]

        if mask is not None:
            mask = transforms.ToTensor()(mask).unsqueeze(0)

        normalized_entry = {
            'image': image,
            'target': {
                'boxes': torch.tensor([bbox], dtype=torch.float32),
                'labels': torch.tensor([label], dtype=torch.int64),
                'masks': mask
            }
        }

        normalized_dataset.append(normalized_entry)

    return normalized_dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        entry = self.dataset_list[idx]
        image = Image.open(entry['path']).convert("RGB")
        bbox = list(map(float, entry['bbox'].split()))  # Convert to float
        label = entry['label']
        mask = Image.open(entry['mask_path']).convert("L") if entry['mask_path'] else None

        if self.transform:
            if mask is not None:
                image, mask = self.transform(image, mask)  # Pass both image and mask
            else:
                image = self.transform(image)

        target = {
            'boxes': torch.tensor([bbox], dtype=torch.float32),
            'labels': torch.tensor([label], dtype=torch.int64),
            'masks': torch.tensor(mask, dtype=torch.float32).unsqueeze(0) if mask is not None else None
        }

        return image, target


image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CombinedTransform:
    def __call__(self, image, mask):
        image = image_transform(image)
        if mask is not None:
            mask = mask_transform(mask)
        return image, mask

import cv2
import matplotlib.pyplot as plt

def show_images_from_dict(updated_dict):
    for entry in updated_dict:
        image_path = entry['path']
        bbox = list(map(int, entry['bbox'].split()))  # Assuming bbox is stored as a string
        label = entry['label']
        mask_path = entry['mask_path']
        image = cv2.imread(image_path)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(image, str(label), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image with Bounding Box")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Segmentation Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.show()

def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue', marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

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
#show_images_from_dict(updated_dict)


def plot_bounding_boxes(dataset):
    plt.figure(figsize=(10, 6))

    for entry in dataset:
        image = entry['image'].permute(1, 2, 0).numpy()  # Change shape for plotting
        boxes = entry['target']['boxes'].numpy()

        plt.imshow(image)

        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            plt.scatter(center_x, center_y, color='red')  # Plot center point

    plt.title('Scatter Plot of Bounding Box Centers')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.axis('off')
    plt.show()

def split_dataset(dataset, train_size=0.8):
    dataset_list = dataset.dataset_list
    random.shuffle(dataset_list)
    split_index = int(len(dataset_list) * train_size)
    train_data = dataset_list[:split_index]
    val_data = dataset_list[split_index:]
    return train_data, val_data


dataset = CustomDataset(updated_dict, transform=CombinedTransform())

train_data, val_data = split_dataset(dataset)
print("This is the train data:")
print(train_data)
print("This is the validation data:")
print(val_data)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)


def print_dataloader_contents(dataloader):
    for batch_idx, data in enumerate(dataloader):
        print("  Image " , batch_idx)
        print(data)
        print(" ")
print_dataloader_contents(train_dataloader)

"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
import torchvision
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)

count = 1
num_epochs =  80
epoch =  num_epochs
listnew = []
for i in range(num_epochs):
  epoch_loss = 0
  model.train()
  for data_entry in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
      image_path = data_entry['path'][0]
      image = Image.open(image_path).convert("RGB")
      image = image_transform(image).to(device)
      mask_path = data_entry['mask_path'][0]
      mask = Image.open(mask_path).convert("L")
      mask = mask_transform(mask).to(device)
      bbox = torch.tensor([[float(coord) for coord in data_entry['bbox'][0].split()]], dtype=torch.float32).to(device)
      label = data_entry['label'].to(device)
      target = {
          'boxes': bbox,
          'labels': label,
          'masks': mask
      }

      optimizer.zero_grad()
      loss_dict = model([image], [target])
      losses = sum(loss for loss in loss_dict.values())
      epoch_loss += losses.item()
      losses.backward()
      optimizer.step()
  listnew.append(epoch_loss)
  print("loss for ",count," is ",epoch_loss)
  count  = count + 1

print("listnew is ",listnew)
plot_training_loss(listnew)

"""

import torch
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

l1_lambda = 1e-5
patience = 10  
best_loss = float('inf')
epochs_without_improvement = 0

num_epochs = 80
listnew = []

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for data_entry in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        image_path = data_entry['path'][0]
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image).to(device)

        mask_path = data_entry['mask_path'][0]
        mask = Image.open(mask_path).convert("L")
        mask = mask_transform(mask).to(device)

        bbox = torch.tensor([[float(coord) for coord in data_entry['bbox'][0].split()]], dtype=torch.float32).to(device)
        label = data_entry['label'].to(device)

        target = {
            'boxes': bbox,
            'labels': label,
            'masks': mask
        }

        optimizer.zero_grad()
        loss_dict = model([image], [target])
        losses = sum(loss for loss in loss_dict.values())

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        losses += l1_lambda * l1_norm  # Add L1 loss

        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += losses.item()

    # Log the epoch loss
    listnew.append(epoch_loss)
    print(f"Loss for epoch {epoch + 1}: {epoch_loss:.2f}")

    # Early stopping logic
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_without_improvement = 0
        model_save_path = '/content/mask_rcnn_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch + 1}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break  # Stop training

print("Loss history:", listnew)
plot_training_loss(listnew)


model_save_path = '/content/mask_rcnn_model.pth'

torch.save(model.state_dict(), model_save_path)

print(f'Model saved at {model_save_path}')

test =r'/content/tester/tester'
import os
testerfile = []

for file_name in os.listdir(test):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        full_path = os.path.join(test, file_name)

        testerfile.append(full_path)

print("This is the list that has the file for the mitotic testing")
print(testerfile)



import torch
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image):
    input_tensor = torch.tensor(image).permute(2, 0, 1)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor.float() / 255.0

def run_inference(model, image):
    model.eval()
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        detections = model(input_tensor)
    return detections

def plot_mitotic_detections(image, boxes, scores, threshold=0.5):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()

    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            plt.text(x1, y1, f'Score: {score.item():.2f}', color='white', fontsize=12, backgroundcolor='red')

    plt.axis('off')
    plt.title('Mitotic Detections', fontsize=16)
    plt.tight_layout()
    plt.show()

class_names = ['Mitotic']
threshold = 0.5

for img_path in testerfile:
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = run_inference(model, image_rgb)

    for i in detections:
        boxes = i["boxes"]
        labels = i["labels"]
        scores = i["scores"]
        mitotic_boxes = []
        mitotic_scores = []

        for box, label, score in zip(boxes, labels, scores):
            if label.item() == 1 and score >= threshold:
                mitotic_boxes.append(box)
                mitotic_scores.append(score)

        mitotic_boxes = torch.stack(mitotic_boxes) if mitotic_boxes else torch.empty((0, 4))
        mitotic_scores = torch.tensor(mitotic_scores) if mitotic_scores else torch.empty((0,))

        if mitotic_boxes.size(0) > 0:
            print("Mitotic Boxes (Threshold > 0.5):", mitotic_boxes)
            print("Mitotic Scores (Threshold > 0.5):", mitotic_scores)
            plot_mitotic_detections(image_rgb, mitotic_boxes, mitotic_scores, threshold)

print("End of the code for the best")
