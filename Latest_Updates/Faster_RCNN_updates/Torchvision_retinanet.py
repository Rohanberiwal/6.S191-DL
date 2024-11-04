import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
model = retinanet_resnet50_fpn(pretrained=True)
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
                'original_image_path': image_path,  # Save original image path
                'cropped_image_path': cropped_image_path,  # Save cropped image path
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

import random
from PIL import ImageEnhance
import os
import imgaug.augmenters as iaa
import numpy as np

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

print(updated_dict)
mitotic_data = collect_mitotic_data(updated_dict)
print(mitotic_data)

plot_bounding_boxes(mitotic_data, "Bounding Boxes Before Normalization")
normalized_data = normalize_mitotic_data(mitotic_data)
plot_bounding_boxes(normalized_data, "Bounding Boxes After Normalization")

from PIL import Image

class CustomMitoticDataset(Dataset):
    def __init__(self, mitotic_data, transforms=None):
        self.mitotic_data = mitotic_data
        self.transforms = transforms

    def __len__(self):
        return len(self.mitotic_data)

    def __getitem__(self, idx):
        data = self.mitotic_data[idx]
        cropped_image_path = data['cropped_image_path']
        box = data['bounding_box']
        label = data['label']

        image = Image.open(cropped_image_path).convert("RGB")
        original_size = image.size  # This should give (width, height)

        if self.transforms is not None:
            image = self.transforms(image)

        # The image tensor will have shape (C, H, W) after transformations
        new_size = image.shape[1:]  # Get (height, width)

        # Resize the bounding box after the image transformation
        box = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
        box = resize_bbox(box, original_size, (new_size[1], new_size[0]))  # (width, height) for resize_bbox

        label = torch.tensor([label], dtype=torch.int64)

        return image, {'boxes': box, 'labels': label}


import torchvision.transforms as T
height = 224
width = 224
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize((height, width)))  # Set your desired height and width
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def resize_bbox(bbox, original_size, new_size):
    original_width, original_height = original_size
    new_width, new_height = new_size
    bbox[:, 0] = bbox[:, 0] * (new_width / original_width)  # xmin
    bbox[:, 1] = bbox[:, 1] * (new_height / original_height)  # ymin
    bbox[:, 2] = bbox[:, 2] * (new_width / original_width)  # xmax
    bbox[:, 3] = bbox[:, 3] * (new_height / original_height)  # ymax
    return bbox
total_size = len(mitotic_data)
train_size = int(total_size * 1)
train_dataset = CustomMitoticDataset(mitotic_data, transforms=get_transform(train=True))



from torch.utils.data import DataLoader
trainer_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


def convert_boxes_format(boxes):
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
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        if not (x_max > x_min and y_max > y_min):
            print(f"Invalid box: {box}")
        assert x_max > x_min and y_max > y_min, f"Invalid box {box}"

def inspect_data_loader(data_loader, num_batches=1):
    for batch_idx, (images, targets) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break

        print(f"Batch {batch_idx + 1}:")
        print(f"Number of images: {len(images)}")

        # Print the shape of the images
        for i, img in enumerate(images):
            print(f"Image {i + 1} shape: {img.shape}")

        # Print the targets
        for i, target in enumerate(targets):
            boxes = target['boxes'] if 'boxes' in target else None
            labels = target['labels'] if 'labels' in target else None
            print(f"Target {i + 1}:")
            if boxes is not None:
                print(f"  Boxes: {boxes.shape} -> {boxes}")
            if labels is not None:
                print(f"  Labels: {labels.shape} -> {labels}")


inspect_data_loader(trainer_loader, num_batches=2)


num_epochs =  300
learning_rate = 1e-5
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.937, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def train_one_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        for target in targets:
            boxes = target['boxes']
            boxes = convert_boxes_format(boxes)
            target['boxes'] = boxes
            validate_boxes(boxes)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()
    if scheduler:
        scheduler.step()

    return total_loss / len(data_loader)

training_losses = []

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, trainer_loader, optimizer, device, scheduler)
    training_losses.append(train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), training_losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.xticks(range(1, num_epochs + 1))
plt.show()

model_save_path = '/content/Faster_rcnn_model.pth'

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
