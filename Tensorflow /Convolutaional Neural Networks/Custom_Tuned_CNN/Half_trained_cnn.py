import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches

standard_dict_mitotic = {}
standard_dict_non_mitotic = {}

def extract_and_preprocess_patches(image_dict):
    X_patches = []
    y_labels = []
    
    for image_path, boxes in image_dict.items():
        try:
            # Load the image
            image = Image.open(image_path)
            
            for xmin, ymin, width, height in boxes:
                # Extract patch based on bounding box coordinates
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                
                # Resize patch to match model input size (e.g., 224x224) and convert to numpy array
                patch = patch.resize((224, 224))
                patch = np.array(patch) / 255.0  # Normalize pixel values
                
                X_patches.append(patch)
                
                # Determine label based on the presence of 'mitotic' in image path
                if 'mitotic' in image_path.lower():
                    y_labels.append(1)  # Mitotic
                else:
                    y_labels.append(0)  # Non-mitotic
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    return np.array(X_patches), np.array(y_labels)

def plot_patches(image_path, boxes):
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        # Plot each bounding box as a small patch
        for xmin, ymin, width, height in boxes:
            patch = image.crop((xmin, ymin, xmin + width, ymin + height))
            patch = patch.resize((100, 100))  # Resize for display
            ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting patches for image {image_path}: {e}")

def plot_boxes_together(dict_mitotic, dict_non_mitotic):
    # Iterate over keys that are common in both dictionaries
    common_keys = set(dict_mitotic.keys()).intersection(dict_non_mitotic.keys())
    
    for key in common_keys:
        try:
            # Load the image
            image = Image.open(key)
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            ax = plt.gca()

            # Plot mitotic bounding boxes as small patches
            for xmin, ymin, width, height in dict_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))  # Resize for display
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # Plot non-mitotic bounding boxes as small patches
            for xmin, ymin, width, height in dict_non_mitotic[key]:
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                patch = patch.resize((100, 100))  # Resize for display
                ax.imshow(patch, extent=(xmin, xmin + width, ymin, ymin + height), alpha=0.6)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)

            plt.title(f"Image: {os.path.basename(key)}")
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error processing image {key}: {e}")

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

def print_mitotic(json_mitotic):
    print("This is the mitotic printer function")
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
            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        print("------------------------")
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  
    return standard_dict_mitotic

def print_filename_bbox(json_file):
    print("This is the printer filename function for non-mitotic")
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
            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        print("------------------------")
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  # Store boundary boxes directly
    return standard_dict_non_mitotic

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
                boxes.append([x, y, width, height])  # Append the coordinates directly
            target_dict[img_path] = boxes
    
    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")

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

def save_patches(image_dict, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    patch_count = 0
    
    for image_path, boxes in image_dict.items():
        try:
            # Load the image
            image = Image.open(image_path)
            
            for xmin, ymin, width, height in boxes:
                # Extract patch based on bounding box coordinates
                patch = image.crop((xmin, ymin, xmin + width, ymin + height))
                
                # Resize patch to match model input size (e.g., 224x224)
                patch = patch.resize((224, 224))
                
                # Save the patch
                patch_save_path = os.path.join(save_dir, f"{prefix}_patch_{patch_count}.jpg")
                patch.save(patch_save_path)
                
                patch_count += 1
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Define root directory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Define directories for saving patches
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\MitoticFolder"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\NonMitoticFolder"

mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
mitotic_output = print_mitotic(mitotic_annotation_file)
non_mitotic_output = print_filename_bbox(non_mitotic_annotation_file)

modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)

load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

for key in standard_dict_mitotic.keys():
    plot_patches(key, standard_dict_mitotic[key])

for key in standard_dict_non_mitotic.keys():
    plot_patches(key, standard_dict_non_mitotic[key])

plot_boxes_together(standard_dict_mitotic, standard_dict_non_mitotic)
path_one  = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
save_patches(standard_dict_mitotic, path_one , "mitotic")
path_two = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"
save_patches(standard_dict_non_mitotic, path_two, "non_mitotic")
print("Done")
