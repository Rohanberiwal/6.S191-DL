import os
import json
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, root, mitotic_annotation_file, non_mitotic_annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(mitotic_annotation_file) as f:
            self.mitotic_annotations = json.load(f)
        with open(non_mitotic_annotation_file) as f:
            self.non_mitotic_annotations = json.load(f)
        
        self.mitotic_image_files = list(self.mitotic_annotations.keys())
        self.non_mitotic_image_files = list(self.non_mitotic_annotations.keys())

    def __len__(self):
        return len(self.mitotic_image_files) + len(self.non_mitotic_image_files)

    def __getitem__(self, idx):
        if idx < len(self.mitotic_image_files):
            img_name = self.mitotic_image_files[idx]
            img_path = os.path.join(self.root, img_name)
            annotation = self.mitotic_annotations[img_name]
            label = 0  # Mitotic label (class 0)
        else:
            img_idx = idx - len(self.mitotic_image_files)
            img_name = self.non_mitotic_image_files[img_idx]
            img_path = os.path.join(self.root, img_name)
            annotation = self.non_mitotic_annotations[img_name]
            label = 1  # Non-mitotic label (class 1)
        
        # Adjust for file extension .jpeg instead of .jpg
        img_path = img_path.replace('.jpg', '.jpeg')

        img = Image.open(img_path).convert("RGB")
        boxes = []
        for region in annotation['regions']:
            shape_attr = region['shape_attributes']
            x = shape_attr['x']
            y = shape_attr['y']
            width = shape_attr['width']
            height = shape_attr['height']
            boxes.append([x, y, x + width, y + height])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target



# Define the transformations
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def extract_filenames_from_json(json_file, root):
    with open(json_file) as f:
        data = json.load(f)
    
    filename_list = []
    for filename, attributes in data.items():
        img_name = attributes['filename']  # Extract the filename from the JSON attributes
        img_path = os.path.join(root, img_name)  # Construct the full image path
        filename_list.append(img_path)
        
    return filename_list

def display_images(image_paths):
    for img_path in image_paths:
        print(f"Displaying image: {img_path}")
        img = Image.open(img_path)
        img.show()




root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file , root)
non_mitotic_filenames = extract_filenames_from_json(non_mitotic_annotation_file , root)

print("Mitotic Filenames:")
print(mitotic_filenames)

print("\nNon-Mitotic Filenames:")
print(non_mitotic_filenames)

for img_path in mitotic_filenames :
        print(f"Displaying image: {img_path}")
        img = Image.open(img_path)
        img.show()


# Dataset
dataset = CustomDataset(root, mitotic_annotation_file, non_mitotic_annotation_file, transforms=get_transform(train=True))


# Split the dataset into train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
split_ratio = 0.8
split = int(split_ratio * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, indices[:split])
test_dataset = torch.utils.data.Subset(dataset, indices[split:])

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 2 classes (mitotic, non-mitotic) + background

# Replace the classifier with a new one, that has num_classes which is user-defined
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Note that this is a minimal example of using fasterrcnn_resnet50_fpn
# This model will have to be custom trained if it is to be used in a specific domain
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# Define the optimizer and learning rate scheduler (already defined in your code)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the number of epochs
num_epochs = 10

# Move model to the appropriate device (GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
