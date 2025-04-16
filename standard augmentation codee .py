import json
import cv2
import os
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np

json_file_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\datasetforbraincancer.json"

import os
import cv2
import matplotlib.pyplot as plt

import os

labelled_dir = r"C:\Users\rohan\OneDrive\Desktop\labelled"

for filename in os.listdir(labelled_dir):
    if "aug" in filename.lower():
        file_path = os.path.join(labelled_dir, filename)
        print("Removing augmented file:", file_path)
        os.remove(file_path)

def get_unlabeled_images(unlabeled_folder):
    unlabeled_image_paths = []

    for file in os.listdir(unlabeled_folder):
        if file.endswith(".png"):
            img_path = os.path.join(unlabeled_folder, file)
            unlabeled_image_paths.append(img_path)

    return unlabeled_image_paths

def get_labeled_images(labeled_folder):
    labeled_image_paths = []

    for file in os.listdir(labeled_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            labeled_image_paths.append(os.path.join(labeled_folder, file))

    return labeled_image_paths

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values in Dataset:")
    print(missing_values[missing_values > 0])
    return missing_values


def show_unlabeled_images(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")
        plt.show()

unlabeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\Unlabeled"
labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"

unlabeled_images = get_unlabeled_images(unlabeled_folder_path)
labeled_images = get_labeled_images(labeled_folder_path)

print("Unlabeled Image Paths:")
print(unlabeled_images)

print("\nLabeled Image Paths:")
print(labeled_images)
with open(json_file_path, "r") as file:
    json_data = json.load(file)

import os
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_labeled_images(labeled_folder):
    labeled_image_paths = []
    for file in os.listdir(labeled_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            labeled_image_paths.append(os.path.join(labeled_folder, file))
    return labeled_image_paths

def extract_case_number(filename):
    return filename.split(".")[0].strip()

def create_json_from_csv(labeled_images, csv_path, output_json):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    dataset = {}
    
    for img_path in labeled_images:
        filename = os.path.basename(img_path)
        case_number = extract_case_number(filename)
        
        for index, row in df.iterrows():
            case_numbers = str(row["Case Number"]).split("\n")  
            if case_number in case_numbers:
                dataset[case_number] = row.to_dict()
                break
    
    with open(output_json, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

def show_images(image_paths):
    shown_images = {}
    for img_path in image_paths:
        if img_path in shown_images:
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")
        plt.show()
        
        shown_images[img_path] = "Yes, shown"
    
    print("Image Display Status:", shown_images)

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
csv_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"
output_json = r"C:\Users\rohan\OneDrive\Desktop\Codes\datasetoutputer.json"

labeled_images = get_labeled_images(labeled_folder_path)
create_json_from_csv(labeled_images, csv_path, output_json)
print(f"Dataset saved to {output_json}")
#show_images(labeled_images)
import os
import re
import pandas as pd

def extract_case_number(filename):
    match = re.match(r"(IN Brain-\d+)", filename)
    return match.group(1) if match else filename

def create_file_to_grade_dict(labeled_images, csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    file_to_grade = {}

    for img_path in labeled_images:
        filename = os.path.basename(img_path)
        case_number = extract_case_number(filename)

        for index, row in df.iterrows():
            csv_case_numbers = str(row["Case Number"]).split("\n")
            csv_case_numbers = [extract_case_number(cn.strip()) for cn in csv_case_numbers]

            if case_number in csv_case_numbers:
                file_to_grade[img_path] = row.get("WHO Grade", "Unknown")
                break
        else:
            file_to_grade[img_path] = "Not Found"

    return file_to_grade

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
csv_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"

labeled_images = [os.path.join(labeled_folder_path, file) for file in os.listdir(labeled_folder_path) if file.endswith((".png", ".jpg"))]

file_to_grade_dict = create_file_to_grade_dict(labeled_images, csv_path)

print("Mapping of file paths to WHO grade:")
for file, grade in file_to_grade_dict.items():
    print(f"{file}: Grade {grade}")


import os
import shutil

output_dir = r"C:\Users\rohan\OneDrive\Desktop\augmented_images"

def clean_output_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Deletes the entire folder and its contents
    os.makedirs(directory, exist_ok=True)  # Recreate an empty folder

clean_output_dir(output_dir)
print(f"Cleaned {output_dir}, ready for augmentation.")
import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from collections import Counter

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3)
])

output_dir = r"C:\Users\rohan\OneDrive\Desktop\augmented_images"

grade_counts = Counter(file_to_grade_dict.values())

plt.figure(figsize=(8, 5))
plt.bar(grade_counts.keys(), grade_counts.values(), color='blue')
plt.xlabel("WHO Grade")
plt.ylabel("Number of Images")
plt.title("Initial Image Distribution")
plt.show()

augmented_file_to_grade_dict = {}

for file, grade in file_to_grade_dict.items():
    img = cv2.imread(file)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    grade_folder = os.path.join(output_dir, f"Grade_{grade}")
    os.makedirs(grade_folder, exist_ok=True)

    original_path = os.path.join(grade_folder, os.path.basename(file))
    cv2.imwrite(original_path, img)

    augmented_file_to_grade_dict[original_path] = grade

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    for i in range(5):
        augmented = transform(image=img)["image"]
        aug_path = os.path.join(grade_folder, f"aug_{i}_{os.path.basename(file)}")
        cv2.imwrite(aug_path, augmented)

        augmented_file_to_grade_dict[aug_path] = grade

        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
        plt.title(f"Augmented {i+1}")
        plt.axis("off")

    plt.show()

augmented_counts = {
    grade_folder: len(os.listdir(os.path.join(output_dir, grade_folder)))
    for grade_folder in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, grade_folder))
}

plt.figure(figsize=(8, 5))
plt.bar(augmented_counts.keys(), augmented_counts.values(), color='green')
plt.xlabel("WHO Grade")
plt.ylabel("Number of Images")
plt.title("Image Distribution After Augmentation")
plt.xticks(rotation=45)
plt.show()
