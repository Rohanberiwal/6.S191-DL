
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
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


import matplotlib.pyplot as plt
from PIL import Image

from PIL import Image
import os


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


import os

def generate_coco_format_with_label(input_dict):
    output = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "mitotic_region"
        }]
    }
    
    annotation_id = 1
    image_id = 1
    
    for image_path, bboxes in input_dict.items():
        image_name = os.path.basename(image_path)
        image_height = 800
        image_width = 600
        
        image_entry = {
            "id": image_id,
            "file_name": image_name,
            "height": image_height,
            "width": image_width
        }
        output["images"].append(image_entry)
        
        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            area = width * height
            
            annotation = {
                "image_id": image_id,
                "label": 1,
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "iscrowd": 0
            }
            output["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    return output

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

output_data =  generate_coco_format_with_label(updated_dict)
print(output_data)

import json
print(json.dumps(output_data, indent=4))


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
save_json(output_data, filename = 'output_coco_json')




import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Helper function to create the mask
def create_mask(image, bbox):
    width, height = image.size
    mask = torch.zeros((height, width), dtype=torch.uint8)
    x_min, y_min, x_max, y_max = bbox
    mask[y_min:y_max, x_min:x_max] = 1
    return mask

def prepare_detr_dataset(output, split_ratio=0.8, image_dir="/content/Train_mitotic/Train_mitotic"):
    train_dataset = {"images": [], "annotations": []}
    val_dataset = {"images": [], "annotations": []}

    annotations_id = 1
    image_id = 1
    
    categories = output["categories"]
    category_map = {category["id"]: category["name"] for category in categories}
    
    total_images = output["images"]
    random.shuffle(total_images)
    num_train_images = int(len(total_images) * split_ratio)
    train_images = total_images[:num_train_images]
    val_images = total_images[num_train_images:]

    for image_entry in train_images:
        image_name = image_entry["file_name"]
        image_height = image_entry["height"]
        image_width = image_entry["width"]
        
        train_dataset["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": image_height,
            "width": image_width
        })
        
        # Full path to the image file
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # Open the image

        for annotation in output["annotations"]:
            if annotation["image_id"] == image_id:
                x_min, y_min, width, height = annotation["bbox"]
                x_max = x_min + width
                y_max = y_min + height
                category_id = annotation["label"]

                # Create the mask for the current object using the bounding box
                mask = create_mask(image, [x_min, y_min, x_max, y_max])
                mask_path = f"/content/mask_image_{image_id}_{annotations_id}.png"
                mask_image = Image.fromarray(mask.numpy() * 255)  # Convert mask to image (0-255)
                mask_image.save(mask_path)  # Save the mask image

                # Display the image and its mask
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(mask_image, cmap='gray')
                plt.title("Generated Mask")
                plt.axis('off')

                plt.show()

                train_dataset["annotations"].append({
                    "id": annotations_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                    "mask_path": mask_path  # Store the mask path
                })
                annotations_id += 1
        
        image_id += 1

    image_id = 1
    annotations_id = 1
    for image_entry in val_images:
        image_name = image_entry["file_name"]
        image_height = image_entry["height"]
        image_width = image_entry["width"]
        
        val_dataset["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": image_height,
            "width": image_width
        })
        
        # Full path to the image file
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # Open the image

        for annotation in output["annotations"]:
            if annotation["image_id"] == image_id:
                x_min, y_min, width, height = annotation["bbox"]
                x_max = x_min + width
                y_max = y_min + height
                category_id = annotation["label"]

                # Create the mask for the current object using the bounding box
                mask = create_mask(image, [x_min, y_min, x_max, y_max])
                mask_path = f"/content/mask_image_{image_id}_{annotations_id}.png"
                mask_image = Image.fromarray(mask.numpy() * 255)  # Convert mask to image (0-255)
                mask_image.save(mask_path)  # Save the mask image

                # Display the image and its mask
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(mask_image, cmap='gray')
                plt.title("Generated Mask")
                plt.axis('off')

                plt.show()

                val_dataset["annotations"].append({
                    "id": annotations_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                    "mask_path": mask_path  # Store the mask path
                })
                annotations_id += 1
        
        image_id += 1

    return train_dataset, val_dataset


train_dataset , val_dataset = prepare_detr_dataset(output_data, split_ratio=0.8)
print(train_dataset)
print(val_dataset)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
"""
class DetrDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, idx):
        image_info = self.dataset['images'][idx]
        image_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        target = {
            'boxes': [],
            'labels': [],
            'masks': [],
            'area': [],
            'iscrowd': [],
        }

        for annotation in self.dataset['annotations']:
            if annotation['image_id'] == image_info['id']:
                x_min, y_min, width, height = annotation['bbox']
                x_max = x_min + width
                y_max = y_min + height
                target['boxes'].append([x_min, y_min, x_max, y_max])
                target['labels'].append(annotation['category_id'])
                target['area'].append(annotation['area'])
                target['iscrowd'].append(annotation['iscrowd'])

                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 1
                target['masks'].append(mask)

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['masks'] = torch.tensor(target['masks'], dtype=torch.uint8)
        target['area'] = torch.tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, target
"""


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DetrDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, idx):
        image_info = self.dataset['images'][idx]
        image_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        target = {
            'boxes': [],
            'labels': [],
            'masks': [],
            'area': [],
            'iscrowd': [],
        }

        for annotation in self.dataset['annotations']:
            if annotation['image_id'] == image_info['id']:
                x_min, y_min, width, height = annotation['bbox']
                x_max = x_min + width
                y_max = y_min + height

                target['boxes'].append([x_min, y_min, x_max, y_max])
                target['labels'].append(annotation['category_id'])
                target['area'].append(annotation['area'])
                target['iscrowd'].append(annotation['iscrowd'])

                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 1
                target['masks'].append(mask)

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['masks'] = torch.tensor(target['masks'], dtype=torch.uint8)
        target['area'] = torch.tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(np.array(image)).float().permute(2, 0, 1)
        image = image.unsqueeze(0)

        return image, target


def create_dataloader(train_dataset, val_dataset, image_dir, batch_size=4, transform=None):
    train_data = DetrDataset(train_dataset, image_dir, transform=transform)
    val_data = DetrDataset(val_dataset, image_dir, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_dataloader, val_dataloader

train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset, image_dir="/content/Train_mitotic/Train_mitotic", batch_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

model.to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        print("Images in the format of tensor:")
        print(images)  # Print out the tensor shape for debugging
        
        # Move targets to device (e.g., GPU)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 
        
        # Loop through each target dictionary in the batch and print the required information
        for target in targets:
            print("Labels:", target['labels'])
            print("Boxes:", target['boxes'])
            print("Masks:", target['masks'])
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, labels=targets)  # Ensure images is a tensor and labels are targets
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss
train(model, train_dataloader, optimizer, device)
