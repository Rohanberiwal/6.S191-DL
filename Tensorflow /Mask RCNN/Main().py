import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet152
import matplotlib.pyplot as plt
import subprocess
import platform
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
import torchvision.transforms as T
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}
import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
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
        masks = []

        if img_path in self.mitotic_dict:
            for box in self.mitotic_dict[img_path]:
                boxes.append(box)
                labels.append(1)  # Mitotic label
                mask = self.create_mask(box, img.size)
                masks.append(mask)

        if img_path in self.non_mitotic_dict:
            for box in self.non_mitotic_dict[img_path]:
                boxes.append(box)
                labels.append(0)  # Non-mitotic label
                mask = self.create_mask(box, img.size)
                masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Filter out invalid boxes
        valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_indices]
        labels = labels[valid_indices]
        masks = masks[valid_indices]

        if self.transforms:
            img = self.transforms(img)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)

        return img, target

    def create_mask(self, box, img_size):
        mask = Image.new('L', img_size, 0)
        draw = ImageDraw.Draw(mask)

        # Drawing the rectangle based on box coordinates
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid box coordinates: {box}")

        draw.rectangle([x0, y0, x1, y1], outline=1, fill=1)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np)

        return mask_tensor



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

print(standard_dict_mitotic)
print(standard_dict_non_mitotic)
print("end")





import numpy as np
from PIL import Image
import os

print("This   is the mask generator fucntion")
import os
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
from tqdm import tqdm

# Custom Dataset Class
class MitoticDataset(Dataset):
    def __init__(self, image_bbox_dict, transform=None):
        self.image_bbox_dict = image_bbox_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_bbox_dict)

    def __getitem__(self, idx):
        image_path = list(self.image_bbox_dict.keys())[idx]
        bounding_boxes = self.image_bbox_dict[image_path]
        
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        mask = create_binary_mask(image_path, bounding_boxes)
        
        # Convert to PIL image
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        image = F.to_tensor(image)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)
        
        # Create target dictionary
        target = {
            'boxes': torch.tensor(bounding_boxes, dtype=torch.float32),
            'labels': torch.ones(len(bounding_boxes), dtype=torch.int64),
            'masks': mask.unsqueeze(0)  # Add a batch dimension
        }
        
        return image, target

def create_binary_mask(image_path, bounding_boxes):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for bbox in bounding_boxes:
        x, y, w, h = bbox
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        mask[y_min:y_max, x_min:x_max] = 1

    return mask

def split_dataset(dataset, train_ratio=0.7):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Perform the split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    metrics = {}
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            for loss_name, loss_value in loss_dict.items():
                metrics[loss_name] = metrics.get(loss_name, 0) + loss_value.item()
        
        # Compute average loss over all batches
        for key in metrics:
            metrics[key] = metrics[key] / len(data_loader)
        
    return metrics

# Initialize the model, optimizer, and device
num_classes = 2  # Adjust this if you have more classes
model = maskrcnn_resnet50_fpn(pretrained=True)

# Replace the RoIHeads with new heads for classification and segmentation
in_features = model.roi_heads.box_roi_pool.out_channels

# Update the box and mask predictor to accommodate the number of classes
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features,
    num_classes
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)

image_bbox_dict = standard_dict_mitotic  # Your data dictionary
dataset = MitoticDataset(image_bbox_dict)
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.7)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
print("Data loaded successfully")
