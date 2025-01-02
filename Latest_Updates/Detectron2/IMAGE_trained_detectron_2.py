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

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import zipfile
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T

def plot_image(image_dict, title="Images"):
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

root = r"/content/Train_mitotic/Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
input_zip_path =  '/content/Train_mitotic.zip'
output_dir = '/content/Train_mitotic'
extract_zip_file(input_zip_path, output_dir)
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
modify_dict_inplace(standard_dict_mitotic, root)

print(standard_dict_mitotic)

print("Code completed")


print("This is the laod fucntoon of the the code ")
import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import random
from sklearn.model_selection import train_test_split

def get_mitotic_dicts(standard_dict_mitotic, split_ratio=0.7):
    mitotic_dicts = []
    for idx, (image_path, boxes) in enumerate(standard_dict_mitotic.items()):
        record = {}

        filename = image_path
        with Image.open(filename) as img:
            width, height = img.size

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for box in boxes:
            x, y, w, h = box
            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,  # Mitotic class, single class, so category_id is 0
            }
            objs.append(obj)
        record["annotations"] = objs
        mitotic_dicts.append(record)

    # Shuffle the dataset
    random.shuffle(mitotic_dicts)

    # Split the dataset into train and validation
    train_size = int(len(mitotic_dicts) * split_ratio)
    train_dataset = mitotic_dicts[:train_size]
    val_dataset = mitotic_dicts[train_size:]

    return train_dataset, val_dataset

from detectron2.data import DatasetCatalog, MetadataCatalog

train_dataset, val_dataset = get_mitotic_dicts(standard_dict_mitotic)
print(train_dataset, 'train_dataset')

DatasetCatalog.register("mitotic_train_data", lambda: train_dataset)
DatasetCatalog.register("mitotic_val_data", lambda: val_dataset)
MetadataCatalog.get("mitotic_train_data").set(thing_classes=["mitotic"])
MetadataCatalog.get("mitotic_val_data").set(thing_classes=["mitotic"])

print(train_dataset[:2])
print(val_dataset[:2])

print("This is the last line of the record")
print("Congfi")

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
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("mitotic_train_data",)
    cfg.DATASETS.TEST = ("mitotic_val_data",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 500    # Number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # Number of regions per image used to train
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (only mitotic)
    cfg.MODEL.DEVICE = "cpu"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CustomEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self._predictions = []
        self._gt_annotations = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # Extract image_id
            image_id = input['image_id']

            # Process predictions
            instances = output['instances']
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()

            for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                x, y, w, h = box
                self._predictions.append({
                    'image_id': image_id,
                    'category_id': int(cls),
                    'bbox': [float(x), float(y), float(w - x), float(h - y)],
                    'score': float(score)
                })

            # Process ground truth annotations
            gt_annotations = input['annotations']
            for ann in gt_annotations:
                bbox = ann['bbox']
                x, y, w, h = bbox
                self._gt_annotations.append({
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'bbox': [float(x), float(y), float(w - x), float(h - y)]
                })

    def calculate_iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def evaluate(self):

        pred_dict = defaultdict(list)
        gt_dict = defaultdict(list)

        for pred in self._predictions:
            pred_dict[pred['image_id']].append(pred)

        for gt in self._gt_annotations:
            gt_dict[gt['image_id']].append(gt)

        precision_sum = 0
        recall_sum = 0
        num_images = 0
        iou_threshold = 0.5

        for image_id in pred_dict:
            num_images += 1

            preds = pred_dict[image_id]
            gts = gt_dict.get(image_id, [])

            if not gts:
                continue

            matched_gt = [False] * len(gts)
            tp = 0
            fp = 0

            for pred in preds:
                pred_box = pred['bbox']
                pred_class = pred['category_id']
                pred_score = pred['score']

                best_iou = 0
                best_gt_index = -1

                for i, gt in enumerate(gts):
                    if gt['category_id'] == pred_class:
                        gt_box = gt['bbox']
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_index = i

                if best_iou >= iou_threshold and best_gt_index >= 0 and not matched_gt[best_gt_index]:
                    tp += 1
                    matched_gt[best_gt_index] = True
                else:
                    fp += 1

            fn = len(gts) - sum(matched_gt)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_sum += precision
            recall_sum += recall

        average_precision = precision_sum / num_images if num_images > 0 else 0
        average_recall = recall_sum / num_images if num_images > 0 else 0
        f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) > 0 else 0

        print(f"Average Precision: {average_precision:.4f}")
        print(f"Average Recall: {average_recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")


import os

def build_custom_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return CustomEvaluator(dataset_name, output_dir=output_folder)

from detectron2.engine import DefaultTrainer

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_custom_evaluator(cfg, dataset_name, output_folder)


from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset


cfg = setup_cfg()

try:
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print("Training done successfully")
        evaluator = build_custom_evaluator(cfg, "mitotic_val_data")
        val_loader = build_detection_test_loader(cfg, "mitotic_val_data")
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
except RuntimeError as e:
        print(f"Runtime error: {e}")
except TypeError as e:
        print(f"Type error: {e}")
print("Everything good ")
print("Training and validation done successfully with God's grace")
import os
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
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
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
import torch

def setup_predictor(cfg):
    print("Setting up predictor with configuration.")
    predictor = DefaultPredictor(cfg)
    return predictor

def predict_and_visualize(predictor, image_path, metadata):
    print(f"Predicting and visualizing for image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    print("Predictions:")
    for i in range(len(boxes)):
        print(f"Box: {boxes[i].tensor.numpy()}, Score: {scores[i].item()}, Class: {classes[i].item()}")
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = Image.fromarray(v.get_image()[:, :, ::-1])
    result_img.show()

print("Loading trained weights for prediction.")
cfg.MODEL.WEIGHTS = trainer.checkpointer.get_checkpoint_file()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

print("Setting up predictor.")
predictor = setup_predictor(cfg)




import os
from PIL import Image

def list_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

def show_images(file_paths):
    for file_path in file_paths:
        # Check if the file is an image
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                # Open and show the image
                with Image.open(file_path) as img:
                    img.show()
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
        else:
            print(f"{file_path} is not an image file")

directory_path = r"C:\Users\rohan\OneDrive\Desktop\Testing_folder"
file_paths = list_file_paths(directory_path)
print(file_paths)
file_paths.append( r'C:\Users\rohan\OneDrive\Desktop\A00_01.jpg')
count_images = 0
for i in file_paths :
    test_image_path =  i
    metadata = MetadataCatalog.get("mitotic_train_data")

    print("Running prediction and visualization.")
    predict_and_visualize(predictor, test_image_path, metadata)
    print("Completed prediction and visualization.")
    count_images = count_images +1

print("end of the computation")
