import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torchvision.transforms import functional as F
from PIL import Image
# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import json
import os
import numpy as np
from PIL import Image

def json_to_coco(json_file, output_coco_file):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize COCO format dictionary
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Mapping for category IDs
    category_id_map = {}
    
    # Initialize category ID counter
    category_id = 1
    
    # Iterate over each image in JSON file
    image_id = 1
    annotation_id = 1
    
    for image_filename, image_data in data.items():
        # Load image to get width and height
        image_path = image_data['filename']  # Adjust based on your JSON structure
        image = Image.open(image_path)
        width, height = image.size
        
        # Add image information to COCO dictionary
        coco_image = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        coco_data["images"].append(coco_image)
        
        # Process annotations
        regions = image_data.get('regions', [])
        for region in regions:
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', 0)
            y = shape_attributes.get('y', 0)
            width = shape_attributes.get('width', 0)
            height = shape_attributes.get('height', 0)
            
            # Convert to COCO format bbox [x_min, y_min, width, height]
            bbox = [x, y, width, height]
            
            # Extract category label
            category_name = region.get('region_attributes', {}).get('class_name', 'unknown')
            
            # Check if category already exists in category map
            if category_name not in category_id_map:
                category_id_map[category_name] = category_id
                coco_category = {
                    "id": category_id,
                    "name": category_name,
                    "supercategory": "object"
                }
                coco_data["categories"].append(coco_category)
                category_id += 1
            
            # Create annotation dictionary
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_map[category_name],
                "bbox": bbox,
                "area": width * height,  # Area calculation (optional)
                "iscrowd": 0  # 0 for normal annotations, 1 for crowd annotations (optional)
            }
            
            # Add annotation to COCO annotations list
            coco_data["annotations"].append(coco_annotation)
            
            # Increment annotation ID
            annotation_id += 1
        
        # Increment image ID
        image_id += 1
    
    # Write COCO format JSON file
    with open(output_coco_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"Conversion completed. COCO format JSON saved to {output_coco_file}")


def ground_truth_func(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    bounding_boxes = []
    for image_filename, image_data in data.items():
        regions = image_data.get('regions', [])
        for region in regions:
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', None)
            y = shape_attributes.get('y', None)
            width = shape_attributes.get('width', None)
            height = shape_attributes.get('height', None)
            if x is not None and y is not None and width is not None and height is not None:
                bounding_box = [x, y, x + width, y + height]  # Convert to [x_min, y_min, x_max, y_max]
                bounding_boxes.append(bounding_box)
    return bounding_boxes

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
image = Image.open(image_path)

def preprocess_image(image, target_size=800):
    # Convert image to RGB if it's not already in that format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize while maintaining aspect ratio
    width, height = image.size
    if width > height:
        ratio = width / height
        new_width = int(ratio * target_size)
        new_height = target_size
    else:
        ratio = height / width
        new_width = target_size
        new_height = int(ratio * target_size)
    resized_image = image.resize((new_width, new_height))

    # Center crop to target_size
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Convert image to tensor
    image_tensor = F.to_tensor(cropped_image)

    # Normalize image pixel values
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    normalized_image = F.normalize(image_tensor, mean=mean, std=std)

    # Ensure normalized image tensor has a batch dimension
    normalized_image = normalized_image.unsqueeze(0)

    return normalized_image


print("COCO to json")
# Example usage:
image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file= r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
output_coco_file = json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes'

# Preprocess the input image
preprocessed_image = preprocess_image(image)

# Print the preprocessed image tensor
print("Preprocessed Image Tensor Shape:", preprocessed_image.shape)
print("Preprocessed Image Tensor:\n", preprocessed_image)


