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

dataset_list = []

from torch.utils.data import DataLoader, random_split
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
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

def plot_cropped_bounding_boxes(image_dict):
    for image_path, boxes in image_dict.items():
        # Open the image
        img = Image.open(image_path)

        for box in boxes:
            x, y, width, height = box
            # Crop the image using the bounding box
            cropped_img = img.crop((x, y, x + width, y + height))
            
            plt.figure(figsize=(5, 5))
            plt.imshow(cropped_img)
            plt.axis('off')  # Turn off axis
            plt.title(f"Cropped: {image_path}")  # Title with the image path
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
        
        # Open the cropped image instead of the original
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

class AugmentedDataset(Dataset):
    def __init__(self, augmented_data, transform=None):
        self.augmented_data = augmented_data
        self.transform = transform

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        data = self.augmented_data[idx]
        image = Image.open(data['image_path']).convert("RGB")
        bbox = data['bounding_box']
        label = data['label']
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        target = {
            'boxes': torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
            'labels': torch.tensor([label], dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


def collate_fn(batch):
    return tuple(zip(*batch))

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
#plot_cropped_bounding_boxes(updated_dict)
mitotic_data = collect_mitotic_data(updated_dict)
print(mitotic_data)
augmented_data = augment_mitotic_data(mitotic_data, num_augmentations=10000)
print(augmented_data)


dataset = AugmentedDataset(augmented_data, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')


print("end")
