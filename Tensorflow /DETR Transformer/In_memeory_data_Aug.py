import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import math 
import json
from PIL import Image, ImageDraw
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"

NUM_CLASSES = 2 
NUM_OBJECT_QUERIES = 100  
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        if self.norm is not None:
            src = self.norm(src)
        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt

# Object Query Embedding
class ObjectQueryEmbedding(nn.Module):
    def __init__(self, num_queries, d_model):
        super(ObjectQueryEmbedding, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))

    def forward(self, x):
        return self.queries.unsqueeze(1).repeat(1, x.size(1), 1)

# DETR Model
class DETR(nn.Module):
    def __init__(self, num_classes, num_queries, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead),
            num_encoder_layers
        )
        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead),
            num_decoder_layers
        )
        self.object_query_embedding = ObjectQueryEmbedding(num_queries, d_model)
        self.linear_class = nn.Linear(d_model, num_classes + 1)
        self.linear_bbox = nn.Linear(d_model, 4)

    def forward(self, src, mask=None):
        src = self.transformer_encoder(src, mask)
        queries = self.object_query_embedding(src)
        tgt = torch.zeros_like(queries)
        tgt = self.transformer_decoder(tgt, src)
        outputs_class = self.linear_class(tgt)
        outputs_bbox = self.linear_bbox(tgt).sigmoid()
        return outputs_class, outputs_bbox


def print_mitotic(json_mitotic):
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box 

def print_filename_bbox(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  # Store boundary boxes directly


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


def load_annotations_into_dict(annotation_file, root_dir, target_dict):
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        for filename, attributes in data.items():
            img_name = attributes['filename']
            img_path = os.path.join(root_dir, img_name)
            boxes = []
            for region in attributes['regions']:
                shape_attr = region['shape_attributes']
                x = shape_attr['x']
                y = shape_attr['y']
                width = shape_attr['width']
                height = shape_attr['height']
                boxes.append([x, y, width, height])  
            target_dict[img_path] = boxes

    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")



def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

    
    
print_mitotic(mitotic_annotation_file)
print_filename_bbox(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)
list_mitotic = get_file_paths(mitotic_save_dir)
list_non_mitotic = get_file_paths(non_mitotic_save_dir)
print("List of mitotic patches:", list_mitotic)
print("List of non-mitotic patches:", list_non_mitotic)
random.shuffle(list_mitotic)
random.shuffle(list_non_mitotic)

split_index_mitotic = int(0.7 * len(list_mitotic))
split_index_non_mitotic = int(0.7 * len(list_non_mitotic))
train_mitotic = list_mitotic[:split_index_mitotic]
val_mitotic = list_mitotic[split_index_mitotic:]

train_non_mitotic = list_non_mitotic[:split_index_non_mitotic]
val_non_mitotic = list_non_mitotic[split_index_non_mitotic:]

print(f"Training set size (mitotic): {len(train_mitotic)}")
print(f"Validation set size (mitotic): {len(val_mitotic)}")
print(f"Training set size (non-mitotic): {len(train_non_mitotic)}")
print(f"Validation set size (non-mitotic): {len(val_non_mitotic)}")
model = DETR(num_classes=NUM_CLASSES, num_queries=NUM_OBJECT_QUERIES).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()

from sklearn.model_selection import train_test_split


def split_dict(data_dict, split_ratio=0.7):
    keys = list(data_dict.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * split_ratio)
    train_keys = keys[:split_index]
    val_keys = keys[split_index:]
    train_dict = {key: data_dict[key] for key in train_keys}
    val_dict = {key: data_dict[key] for key in val_keys}
    
    return train_dict, val_dict

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def plot_and_store_bboxes(images_with_bboxes, label):
    data = []  
    for image_path, bboxes in images_with_bboxes.items():
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            for bbox in bboxes:
                x, y, w, h = bbox
                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                cropped_img = img.crop((x, y, x + w, y + h))
                data.append({
                    'bbox': bbox,
                    'cropped_img': cropped_img,
                    'label': label
                })
    print(data)
    return data




train_dict, val_dict = split_dict(standard_dict_mitotic)
train_non, val_non = split_dict(standard_dict_non_mitotic)

list_mitotic= plot_and_store_bboxes(train_dict, label=1)
list_non_mitotic = plot_and_store_bboxes(train_non, label=0)
val_data_mitotic = plot_and_store_bboxes(val_dict, label=1)
val_data_non_mitotic = plot_and_store_bboxes(val_non, label=0)
train_data = list_mitotic + list_non_mitotic
val_data = val_data_mitotic + val_data_non_mitotic

print("Phase of the data laodign in the dert ")
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        cropped_img = item['cropped_img']
        bbox = item['bbox']
        label = item['label']
        
        image = F.to_tensor(cropped_img)
        
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, bbox, label


def get_transform():
    return transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images)
    bboxes = [torch.tensor(bbox) for bbox in bboxes]
    labels = [torch.tensor(label) for label in labels]
    return images, bboxes, labels



dataset_train = CustomDataset(train_data, transforms=get_transform())
dataset_validation  = CustomDataset(train_data, transforms=get_transform())
train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset_validation, batch_size=4, shuffle=True, collate_fn=collate_fn)

def print_dataloader_contents(dataloader):
    for batch_idx, (images, bboxes, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Number of images: {len(images)}")
        
        for i in range(len(images)):
            img = images[i]
            print(f"Image {i + 1} - Shape: {img.shape}")
            print(f"Bounding Boxes: {bboxes[i]}")
            print(f"Labels: {labels[i]}")
            print()

        if batch_idx == 1: 
            break


print("Training Loader check")
print_dataloader_contents(train_loader)
print("Validartion check ")
print_dataloader_contents(val_loader)
print("Code ends without any error ")

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),    # Random vertical flip
        transforms.RandomRotation(30),      # Random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        transforms.RandomResizedCrop(size=(800, 800), scale=(0.8, 1.0)),  # Random resized crop
        transforms.ToTensor(),              # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

def augment_dataloader(dataloader, transform):
    augmented_data = []
    for images, bboxes, labels in dataloader:
        augmented_images = []
        for image in images:
            image_pil = transforms.ToPILImage()(image)  
            image_aug = transform(image_pil)  
            augmented_images.append(image_aug) 
        augmented_images = torch.stack(augmented_images)
        augmented_data.append((augmented_images, bboxes, labels))

    return augmented_data

augmentation_transforms = get_augmentation_transforms()
augmented_train_data = augment_dataloader(train_loader, augmentation_transforms)
augmented_val_data = augment_dataloader(val_loader, augmentation_transforms)
print("This is the code")
