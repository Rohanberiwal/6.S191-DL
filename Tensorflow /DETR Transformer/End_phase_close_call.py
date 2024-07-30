
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

import json

def fix_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Content of {file_path}: {content}")  # Debugging line
        
        if not content.strip():
            print("The file is empty.")
            return
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("Attempting to fix the JSON format...")
            
            if not (content.startswith('{') and content.endswith('}')):
                content = '{' + content + '}'
            
            try:
                data = json.loads(content)
                print("Successfully fixed the JSON format.")
            except json.JSONDecodeError as e:
                print(f"Failed to fix JSON format: {e}")
                return
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Successfully fixed and saved JSON data to {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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

def print_json_contents(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            print(json.dumps(data, indent=4))
    
    except FileNotFoundError:
        print(f"The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {json_file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    

fix_json_file(r'C:\Users\rohan\OneDrive\Desktop\Codes\mitotic.json')
print_json_contents(r'C:\Users\rohan\OneDrive\Desktop\Codes\NonMitotic.json')
fix_json_file(r'C:\Users\rohan\OneDrive\Desktop\Codes\NonMitotic.json')

print("Json is fixed ")
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

standard_paths = list(standard_dict_mitotic.keys())
standard_annotations = list(standard_dict_mitotic.values())

non_standard_paths = list(standard_dict_non_mitotic.keys())
non_standard_annotations = list(standard_dict_non_mitotic.values())

standard_train_paths, standard_val_paths, standard_train_annotations, standard_val_annotations = train_test_split(
    standard_paths, standard_annotations, test_size=0.3, random_state=42)

non_standard_train_paths, non_standard_val_paths, non_standard_train_annotations, non_standard_val_annotations = train_test_split(
    non_standard_paths, non_standard_annotations, test_size=0.3, random_state=42)

train_dict = {}
val_dict = {}

for path, annotation in zip(standard_train_paths + non_standard_train_paths, 
                            standard_train_annotations + non_standard_train_annotations):
    train_dict[path] = annotation

for path, annotation in zip(standard_val_paths + non_standard_val_paths, 
                            standard_val_annotations + non_standard_val_annotations):
    val_dict[path] = annotation

print("Training Data:")
for path, annotation in train_dict.items():
    print(f"Path: {path}, Annotation: {annotation}")

print("\nValidation Data:")
for path, annotation in val_dict.items():
    print(f"Path: {path}, Annotation: {annotation}")

print("This is the Fine tune ")
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MitoticDataset(Dataset):
    def __init__(self, root_dir, data_dict, transform=None):
        self.root_dir = root_dir
        self.data_dict = data_dict
        self.transform = transform
        self.data = self.load_annotations()

    def load_annotations(self):
        data = []
        for path, annotation in self.data_dict.items():
            img_name = os.path.basename(path)  # Extract filename from path
            data.append((img_name, annotation))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, boxes = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure image is in PIL format
        
        if self.transform:
            image = self.transform(image)
        
        
        return image, boxes

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    max_targets = max(len(t) for t in targets)
    padded_targets = []
    for t in targets:
        if len(t) < max_targets:
            padding = [[0, 0, 0, 0]] * (max_targets - len(t))
            padded_targets.append(t + padding)
        else:
            padded_targets.append(t)
    targets_tensor = torch.tensor(padded_targets, dtype=torch.float32)
    return images, targets_tensor

train_dataset = MitoticDataset(root, train_dict, transform=transform)
val_dataset = MitoticDataset(root, val_dict, transform=transform)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False , collate_fn=collate_fn)

print("Training Data:")
for image, annotation in train_dataset:
    print(f"Path: {image}, Annotation: {annotation}")

print("\nValidation Data:")
for image, annotation in val_dataset:
    print(f"Path: {image}, Annotation: {annotation}")

print("Data loaded successfully into the dataset.")

def print_dataloader_contents(data_loader, num_batches=1):
    for i, (images, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print("Sample targets:", targets[0])

print_dataloader_contents(train_dataloader, num_batches=2)
print_dataloader_contents(val_dataloader, num_batches=2)

print("This is the train loop")
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
NUM_CLASSES = 2 
NUM_OBJECT_QUERIES = 100  
model = DETR(num_classes=NUM_CLASSES, num_queries=NUM_OBJECT_QUERIES)
model.to(device)

criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

import torch
from torchvision import transforms



import torch
from torchvision import transforms
from tqdm import tqdm

def convert_to_2d_unbatched_tensor(images):
    batch_size, channels, height, width = images.shape
    print(f"Original 4D tensor shape: {images.shape}")  # Print shape of the 4D tensor
    images_2d = images.view(batch_size, channels, -1)  # [batch_size, channels, height * width]
    print(f"After view operation: {images_2d.shape}")  # Print shape after view operation
    images_2d = images_2d.permute(0, 2, 1)  # [batch_size, height * width, channels]
    print(f"After permute operation: {images_2d.shape}")  # Print shape after permute operation
    return images_2d

def preprocess_images(images, target_height=512, target_width=512):
    resize_transform = transforms.Resize((target_height, target_width))
    resized_images = [resize_transform(transforms.ToPILImage()(img)) for img in images]
    return torch.stack([transforms.ToTensor()(img) for img in resized_images])

def train_one_epoch(model, dataloader, optimizer, criterion_class, criterion_bbox, device, scheduler=None):
    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        print("Image is in the tensor" , image)
        
        print("Raw targets:", targets)
        if not isinstance(targets, list):
            raise ValueError("Targets should be a list.")
    
        for i, t in enumerate(targets):
            print(f"Target {i}: {t}")
            if not isinstance(t, dict):
                raise ValueError(f"Each target should be a dictionary, but got {type(t)}")
            if 'boxes' not in t or 'labels' not in t:
                raise ValueError("Each dictionary in targets should contain 'boxes' and 'labels' keys.")
            print(f"Boxes for target {i}: {t['boxes']}")
            print(f"Labels for target {i}: {t['labels']}")
            
    
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor but got {images.dim()}D tensor")
        images_2d = convert_to_2d_unbatched_tensor(images)

        outputs_class, outputs_bbox = model(images_2d)

        if outputs_class.size(-1) != 256:
            raise ValueError(f"Expected embedding dimension of 256 but got {outputs_class.size(-1)}")
        
        loss_class = sum(criterion_class(outputs_class[i].view(-1, outputs_class.size(-1)), targets[i]['labels'].view(-1))
                         for i in range(len(targets)))
        loss_bbox = sum(criterion_bbox(outputs_bbox[i].view(-1, 4), targets[i]['boxes'].view(-1, 4))
                        for i in range(len(targets)))
        total_loss = loss_class + loss_bbox

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return total_loss.item()

def validate(model, dataloader, criterion_class, criterion_bbox, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            print("Image is ")
            print(image)
            
            targets = [{'boxes': torch.tensor(t['boxes'], dtype=torch.float32),
                        'labels': torch.tensor(t['labels'], dtype=torch.int64)} for t in targets]
            
            images_2d = convert_to_2d_unbatched_tensor(images)

            outputs_class, outputs_bbox = model(images_2d)

            loss_class = sum(criterion_class(outputs_class[i].view(-1, outputs_class.size(-1)), targets[i]['labels'].view(-1))
                             for i in range(len(targets)))
            loss_bbox = sum(criterion_bbox(outputs_bbox[i].view(-1, 4), targets[i]['boxes'].view(-1, 4))
                            for i in range(len(targets)))
            total_loss += loss_class.item() + loss_bbox.item()

    return total_loss / len(dataloader)


num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion_class, criterion_bbox, device)
    val_loss = validate(model, val_dataloader, criterion_class, criterion_bbox, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

print("End of the training code")
