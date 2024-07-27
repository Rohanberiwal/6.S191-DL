import json
import os
import detectron2
print(detectron2.__version__)
import sys
print(sys.path)
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def convert_to_coco(json_file, base_dir, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "non_mitotic"},
            {"id": 1, "name": "mitotic"}
        ]
    }
    
    annotation_id = 0
    for image_id, (image_info, image_data) in enumerate(data.items()):
        filename = os.path.join(base_dir, image_data['filename'])
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue
        
        image_entry = {
            "id": image_id,
            "file_name": filename,
            "height": image_data['regions'][0]['shape_attributes']['height'] if image_data['regions'] else 0,
            "width": image_data['regions'][0]['shape_attributes']['width'] if image_data['regions'] else 0
        }
        coco_format["images"].append(image_entry)
        
        for region in image_data['regions']:
            bbox = region['shape_attributes']
            category_id = region['region_attributes'].get('class_id', 0)
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": category_id,
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation_entry)
            annotation_id += 1
    
    with open(output_file, 'w') as f:
        json.dump(coco_format, f)
    print(f"COCO format data saved to {output_file}")

def register_custom_coco_dataset(coco_file, dataset_name):
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
        print(f"Unregistered existing dataset: {dataset_name}")
    
    def load_coco_json(coco_file):
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        dataset_dicts = []
        for image_info in coco_data['images']:
            record = {}
            record["file_name"] = image_info["file_name"]
            record["image_id"] = image_info["id"]
            record["height"] = image_info["height"]
            record["width"] = image_info["width"]
            
            objs = []
            for annotation in coco_data['annotations']:
                if annotation["image_id"] == image_info["id"]:
                    obj = {
                        "bbox": annotation["bbox"],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": annotation["category_id"],
                        "iscrowd": annotation.get("iscrowd", 0)
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
    
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(coco_file))
    MetadataCatalog.get(dataset_name).set(thing_classes=["non_mitotic", "mitotic"])
    print(f"Dataset '{dataset_name}' registered with COCO file: {coco_file}")

def visualize_dataset(dataset_name):
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import random

    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, 3):
        img_path = d["file_name"]
        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(f'Sample from {dataset_name}', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


json_file_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Codes\mitotic.json'
base_dir_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Train_mitotic'
output_coco_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Codes\mitotic_coco.json'

json_file_non_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Codes\NonMitotic.json'
base_dir_non_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Train_mitotic'
output_coco_non_mitotic = r'C:\Users\rohan\OneDrive\Desktop\Codes\non_mitotic_coco.json'

# Convert JSON to COCO format
convert_to_coco(json_file_mitotic, base_dir_mitotic, output_coco_mitotic)
convert_to_coco(json_file_non_mitotic, base_dir_non_mitotic, output_coco_non_mitotic)

# Register datasets
register_custom_coco_dataset(output_coco_mitotic, 'mitotic_data')
register_custom_coco_dataset(output_coco_non_mitotic, 'non_mitotic_data')

# Visualize datasets
print("Visualizing mitotic dataset samples...")
visualize_dataset('mitotic_data')

print("Visualizing non-mitotic dataset samples...")
visualize_dataset('non_mitotic_data')
