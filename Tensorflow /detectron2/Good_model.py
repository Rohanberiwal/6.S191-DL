import json
import os
from PIL import Image
from tqdm import tqdm
import random
def convert_to_coco(json_file, output_file, label_mapping, root_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "non-mitotic"}, {"id": 1, "name": "mitotic"}]
    }

    annotation_id = 1

    for img_id, img_data in tqdm(data.items(), desc="Processing Images"):
        filename = img_data["filename"]
        img_size = img_data["size"]

        # Combine root path with filename
        full_path = os.path.join(root_dir, filename)

        if not os.path.exists(full_path):
            print(f"Warning: File does not exist: {full_path}")
            continue

        with Image.open(full_path) as img:
            width, height = img.size

        # Add image info
        coco_format["images"].append({
            "id": img_id,
            "file_name": full_path,
            "width": width,
            "height": height,
            "size": img_size
        })

        # Add annotations
        for region in img_data["regions"]:
            shape = region["shape_attributes"]
            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]

            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": label_mapping["mitotic"] if "mitotic" in json_file.lower() else label_mapping["non-mitotic"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO format data saved to: {output_file}")


def split_and_combine_coco(mitotic_coco_file, non_mitotic_coco_file, train_output_file, val_output_file, train_ratio=0.7):
    with open(mitotic_coco_file, 'r') as f:
        mitotic_data = json.load(f)

    with open(non_mitotic_coco_file, 'r') as f:
        non_mitotic_data = json.load(f)

    def split_data(data, train_ratio):
        images = data['images']
        annotations = data['annotations']
        
        train_images = []
        val_images = []
        train_annotations = []
        val_annotations = []
        
        random.shuffle(images)
        
        train_size = int(len(images) * train_ratio)
        
        train_image_ids = {img['id'] for img in images[:train_size]}
        val_image_ids = {img['id'] for img in images[train_size:]}
        
        for img in images:
            if img['id'] in train_image_ids:
                train_images.append(img)
            else:
                val_images.append(img)
        
        for ann in annotations:
            if ann['image_id'] in train_image_ids:
                train_annotations.append(ann)
            else:
                val_annotations.append(ann)
        
        return (train_images, train_annotations), (val_images, val_annotations)

    # Split mitotic data
    mitotic_train, mitotic_val = split_data(mitotic_data, train_ratio)
    # Split non-mitotic data
    non_mitotic_train, non_mitotic_val = split_data(non_mitotic_data, train_ratio)

    # Combine mitotic and non-mitotic splits
    train_data = {
        "images": mitotic_train[0] + non_mitotic_train[0],
        "annotations": mitotic_train[1] + non_mitotic_train[1],
        "categories": mitotic_data['categories']
    }
    
    val_data = {
        "images": mitotic_val[0] + non_mitotic_val[0],
        "annotations": mitotic_val[1] + non_mitotic_val[1],
        "categories": mitotic_data['categories']
    }

    # Save combined data to output files
    with open(train_output_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(val_output_file, 'w') as f:
        json.dump(val_data, f, indent=4)

    print(f"Training data saved to: {train_output_file}")
    print(f"Validation data saved to: {val_output_file}")

root_dir = r'C:\Users\rohan\OneDrive\Desktop\Train_mitotic'
mitotic_json_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\mitotic.json'
output_mitotic_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\coco_mitotic.json'
non_mitotic_json_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\NonMitotic.json'
output_non_mitotic_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\coco_non_mitotic.json'
label_mapping = {"mitotic": 1, "non-mitotic": 0}
convert_to_coco(mitotic_json_file, output_mitotic_file, label_mapping, root_dir)
convert_to_coco(non_mitotic_json_file, output_non_mitotic_file, label_mapping, root_dir)

train_output_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\coco_train.json'
val_output_file = r'C:\Users\rohan\OneDrive\Desktop\Codes\coco_val.json'
print("Splitting the data into teh 70  , 30 format ")
split_and_combine_coco(output_mitotic_file, output_non_mitotic_file, train_output_file, val_output_file)
print("splitting done perfectly")
