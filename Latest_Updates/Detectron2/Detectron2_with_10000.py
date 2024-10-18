import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
# Set up logger
import os
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import zipfile
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

import detectron2.data.datasets as d2datasets
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import os


import os
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import random
from PIL import Image
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
from PIL import Image, ImageDraw
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
import os
from PIL import Image
import json
from detectron2.structures import BoxMode
from detectron2.structures import Boxes, Instances
from detectron2.structures import Boxes, Instances
setup_logger()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
sys.setrecursionlimit(2000)  # Increase the recursion limit


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



def plot_image(image_dict, title="Images"):
    """Helper function to plot images and their bounding boxes from a dictionary."""
    for image_path, boxes in image_dict.items():
        try:
            print(f"Trying to open image: {image_path}")
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            for box in boxes:
                x, y, width, height = box

                draw.rectangle([x, y, x + width, y + height], outline="black", width=3)

            img.show(title)

        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue




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


import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


dataset_list = []

from PIL import Image, ImageDraw
import os

def convert_bbox_xywh_to_minmax(bbox):
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def convert_and_save_bounding_boxes(standard_dict, output_img_dir, output_txt_dir):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    for img_path, bboxes in standard_dict.items():
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        img_width, img_height = img.size
        img_filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_filename)

        bbox_labels = []
        image_patches_info = []

        for i, bbox in enumerate(bboxes):
            bbox_minmax = convert_bbox_xywh_to_minmax(bbox)
            x_min, y_min, x_max, y_max = bbox_minmax

            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            cropped_img_filename = f"{base_name}_bbox_{i}.png"
            cropped_img_path = os.path.join(output_img_dir, cropped_img_filename)
            cropped_img.save(cropped_img_path)

            bbox_label = f"{x_min} {y_min} {x_max} {y_max}"
            label = 0  # Ensure this label is consistent with your class index in COCO format
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

    return standard_dict


def load_coco_json(json_file, image_root):
    with open(json_file) as f:
        coco_dict = json.load(f)

    dataset_dicts = []
    for image_info in coco_dict["images"]:
        record = {}
        record["file_name"] = os.path.join(image_root, image_info["file_name"])
        record["image_id"] = image_info["id"]
        record["height"] = image_info["height"]
        record["width"] = image_info["width"]

        # Get annotations for this image
        annotations = [a for a in coco_dict["annotations"] if a["image_id"] == image_info["id"]]
        boxes = []
        category_ids = []

        for anno in annotations:
            boxes.append(anno["bbox"])  # Collect bounding boxes
            category_ids.append(anno["category_id"])  # Collect category IDs

        gt_instances = Instances(image_info["height"], image_info["width"])
        gt_instances.gt_boxes = Boxes(torch.tensor(boxes).float()) if boxes else Boxes(torch.empty((0, 4), dtype=torch.float32))  # Handle empty boxes
        gt_instances.gt_classes = torch.tensor(category_ids) if category_ids else torch.empty((0,), dtype=torch.int64)  # Handle empty classes

        record["gt_instances"] = gt_instances  # Add gt_instances to the record
        dataset_dicts.append(record)

    return dataset_dicts




def convert_to_coco_format(dataset_dict, image_root):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "mitotic"}]
    }

    annotation_id = 1

    for image_id, (image_path, bboxes) in enumerate(dataset_dict.items()):
        image_filename = os.path.basename(image_path)
        full_image_path = os.path.join(image_root, image_filename)
        img = Image.open(full_image_path)
        width, height = img.size

        coco_format["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # mitotic class
                "bbox": [x1, y1, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1

    return coco_format


def save_coco_json(coco_format, output_file):
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)



import json
import random
import copy


from detectron2.evaluation import COCOEvaluator, DatasetEvaluator

def split_coco_dataset(coco_json_path, output_train_json, output_val_json, split_ratio=0.3):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    train_data = copy.deepcopy(coco_data)
    val_data = copy.deepcopy(coco_data)
    image_ids = [image['id'] for image in coco_data['images']]
    random.shuffle(image_ids)
    split_index = int(len(image_ids) * (1 - split_ratio))

    train_image_ids = image_ids[:split_index]
    val_image_ids = image_ids[split_index:]
    def filter_by_image_ids(data, image_ids):
        filtered_data = {
            "images": [image for image in data["images"] if image["id"] in image_ids],
            "annotations": [anno for anno in data["annotations"] if anno["image_id"] in image_ids],
            "categories": data["categories"]
        }
        return filtered_data
    train_data = filter_by_image_ids(train_data, train_image_ids)
    val_data = filter_by_image_ids(val_data, val_image_ids)
    with open(output_train_json, 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(output_val_json, 'w') as f:
        json.dump(val_data, f, indent=4)

def normalize_bboxes(input_dict):
    all_coords = []
    for bboxes in input_dict.values():
        all_coords.extend(bboxes)

    all_coords = np.array(all_coords)

    mean = np.mean(all_coords)
    std_dev = np.std(all_coords)

    normalized_bboxes = {
        key: [(bbox - mean) / std_dev for bbox in bboxes]
        for key, bboxes in input_dict.items()
    }

    return normalized_bboxes

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

print(standard_dict_mitotic)
train_output =  '/content/Faster_RCNN/patch'
train_labeled = '/content/Faster_RCNN/patch_contents'
updated_dict = convert_and_save_bounding_boxes(standard_dict_mitotic, train_output, train_labeled)
normalized_dict = normalize_bboxes(standard_dict_mitotic)

coco_format = convert_to_coco_format(standard_dict_mitotic ,  image_root  = '/content/Train_mitotic/Train_mitotic')
save_coco_json(coco_format, 'output_coco_format.json')

print(coco_format)
split_coco_dataset('output_coco_format.json', 'train_coco_format.json', 'val_coco_format.json')


import json
import os
import random
import copy
import numpy as np
import cv2
from albumentations import (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate, Compose)
from PIL import Image

def augment_and_update_coco(coco_format, image_root, num_augmented_images=1000):
    augmentation = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Rotate(limit=40, p=0.5)
    ])
    
    augmented_coco_format = {
        "images": [],
        "annotations": [],
        "categories": coco_format["categories"]
    }

    annotation_id = len(coco_format['annotations']) + 1 
    augmented_images = 0

    for img_data in coco_format['images']:
        img_path = os.path.join(image_root, img_data['file_name'])
        image = cv2.imread(img_path)
        
        if image is None:
            continue

        for _ in range(num_augmented_images // len(coco_format['images'])):
            augmented = augmentation(image=image)
            augmented_image = augmented['image']
            
            new_file_name = f"aug_{augmented_images}_{img_data['file_name']}"
            augmented_coco_format['images'].append({
                "id": len(augmented_coco_format['images']),
                "file_name": new_file_name,
                "width": augmented_image.shape[1],
                "height": augmented_image.shape[0]
            })
            
            # Update annotations for augmented image
            for annotation in coco_format['annotations']:
                if annotation['image_id'] == img_data['id']:
                    new_bbox = annotation['bbox']  # Adjust if needed for augmentation
                    augmented_coco_format['annotations'].append({
                        "id": annotation_id,
                        "image_id": len(augmented_coco_format['images']) - 1,  # New image ID
                        "category_id": annotation['category_id'],
                        "bbox": new_bbox,
                        "area": new_bbox[2] * new_bbox[3],
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            # Save the augmented image
            cv2.imwrite(os.path.join(image_root, new_file_name), augmented_image)
            augmented_images += 1

    return augmented_coco_format

with open('output_coco_format.json', 'r') as f:
    coco_format = json.load(f)

"""
augmented_coco_format = augment_and_update_coco(coco_format, '/content/Train_mitotic/Train_mitotic', 10000)
save_coco_json(augmented_coco_format, 'augmented_output_coco_format.json')
"""

split_coco_dataset( 'augmented_output_coco_format.json', 'aug_train_coco_format.json', 'aug_val_coco_format.json', split_ratio=0.3)

def register_datasets():
    train_json = '/content/train_coco_format.json'
    val_json = '/content/val_coco_format.json'
    image_root = '/content/Train_mitotic/Train_mitotic'
    DatasetCatalog.register("train_coco_format_new_1", lambda: load_coco_json(train_json, image_root))
    MetadataCatalog.get("train_coco_format_new_1").set(thing_classes=["mitotic"])
    DatasetCatalog.register("val_coco_format_new_1", lambda: load_coco_json(val_json, image_root))
    MetadataCatalog.get("val_coco_format_new_1").set(thing_classes=["mitotic"])

import os
import json
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.utils import comm
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader

def setup_cfg():
   # register_datasets()
    cfg = get_cfg()
    config_path = "/content/faster_rcnn_R_50_FPN_3x.yaml"

    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} not available in the directory!")
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = (3000, 6000)
    cfg.SOLVER.MAX_ITER = 9000
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # Ensure this is a tuple
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = (800,)    # Change this to a tuple
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.IOU_THRESHOLDS = [0.5]
    cfg.DATASETS.TRAIN = ("train_coco_format_new_1",)
    cfg.DATASETS.TEST = ("val_coco_format_new_1",)
    cfg.OUTPUT_DIR = "./output"
    cfg.MODEL.DEVICE = "cpu"

    return cfg

setup_cfg()


from detectron2.structures import Instances

from detectron2.structures import Instances

class CustomInstances(Instances):
    def __init__(self, *args, extra_arg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_arg = extra_arg 

class CustomEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, output_dir=None):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.output_dir = output_dir
        self.metadata = MetadataCatalog.get(dataset_name)
        self._predictions = []
        self._annotations = []

    def reset(self):
        self._predictions = []
        self._annotations = []

    def process(self, inputs, outputs):
      for input_data, output in zip(inputs, outputs):
          image_id = input_data["image_id"]

          if output["instances"].has("pred_boxes"):
              bboxes = output["instances"].pred_boxes.tensor.cpu().numpy()
          else:
              bboxes = np.array([])

          scores = output["instances"].scores.cpu().numpy()
          self._predictions.append({"image_id": image_id, "bboxes": bboxes, "scores": scores})

          if "instances" in input_data:
              # Create a custom instance with an additional argument if needed
              instances = CustomInstances(
                  input_data["instances"].image_size,
                  num_instances=len(input_data["instances"]),
                  extra_arg=your_value  # Replace with your actual value
              )
              self._annotations.append(instances.to("cpu"))
          else:
              self._annotations.append(None) 
              
    def evaluate(self):
        if self.output_dir:
            with open(f"{self.output_dir}/predictions.json", "w") as f:
                json.dump(self._predictions, f)

        coco_evaluator = COCOEvaluator(self.dataset_name, ("bbox",), False, output_dir=self.output_dir)
        coco_evaluator.process(self._annotations)
        metrics = coco_evaluator.evaluate()

        precision = metrics.get("bbox", {}).get("AP", 0)
        recall = metrics.get("bbox", {}).get("AR", 0)

        return {
            "precision": precision,
            "recall": recall,
            **metrics,
        }

cfg = setup_cfg()

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        return CustomEvaluator(dataset_name, cfg, output_dir)

print("Train dataset:", cfg.DATASETS.TRAIN)
print("Validation dataset:", cfg.DATASETS.TEST)
print("Registered datasets:", DatasetCatalog.list())

try :
  trainer = CustomTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()
except Exception as e:
  print(e)
