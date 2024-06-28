import os
import json
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from torchvision.models import resnet50
from torchvision.ops import roi_pool
import torch.nn.functional as F
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import tensorflow as tf
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}
import os
from torchvision.ops import roi_pool
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F 
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

floattype = torch.cuda.FloatTensor

import numpy as np
import torch
# Define TorchROIAlign class
class TorchROIAlign(object):
    
    def __init__(self, output_size, scaling_factor):
        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_align(self, features, scaled_proposal):
        _, num_channels, h, w = features.shape

        xp0, yp0, xp1, yp1 = scaled_proposal
        p_width = xp1 - xp0
        p_height = yp1 - yp0

        w_stride = p_width / self.output_size[1]
        h_stride = p_height / self.output_size[0]

        interp_features = torch.zeros((num_channels, self.output_size[0], self.output_size[1]))

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                x_bin_strt = i * w_stride + xp0
                y_bin_strt = j * h_stride + yp0

                x1 = torch.Tensor([x_bin_strt + 0.25 * w_stride])
                x2 = torch.Tensor([x_bin_strt + 0.75 * w_stride])
                y1 = torch.Tensor([y_bin_strt + 0.25 * h_stride])
                y2 = torch.Tensor([y_bin_strt + 0.75 * h_stride])

                for c in range(num_channels):
                    img = features[0, c]
                    v1 = bilinear_interpolate(img, x1, y1)
                    v2 = bilinear_interpolate(img, x1, y2)
                    v3 = bilinear_interpolate(img, x2, y1)
                    v4 = bilinear_interpolate(img, x2, y2)

                    interp_features[c, j, i] = (v1 + v2 + v3 + v4) / 4

        return interp_features

    def __call__(self, feature_layer, proposals):
        _, num_channels, _, _ = feature_layer.shape

        scaled_proposals = torch.zeros_like(proposals)
        scaled_proposals[:, 0] = proposals[:, 0] * self.scaling_factor
        scaled_proposals[:, 1] = proposals[:, 1] * self.scaling_factor
        scaled_proposals[:, 2] = proposals[:, 2] * self.scaling_factor
        scaled_proposals[:, 3] = proposals[:, 3] * self.scaling_factor

        res = torch.zeros((len(proposals), num_channels, self.output_size[0], self.output_size[1]))
        for idx in range(len(scaled_proposals)):
            proposal = scaled_proposals[idx]
            res[idx] = self._roi_align(feature_layer, proposal)

        return res

# Function to perform bilinear interpolation
def bilinear_interpolate(img, x, y):
    x = x.clamp(min=0, max=img.shape[1] - 1)
    y = y.clamp(min=0, max=img.shape[0] - 1)
    x0 = x.floor().to(torch.int64)
    x1 = (x0 + 1).clamp(max=img.shape[1] - 1)
    y0 = y.floor().to(torch.int64)
    y1 = (y0 + 1).clamp(max=img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    interpolated_values = (Ia * wa) + (Ib * wb) + (Ic * wc) + (Id * wd)
    return interpolated_values

def selective_search(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def get_proposals(image_path):

    image = cv2.imread(image_path)
    proposals = selective_search(image)
    return proposals

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
                bounding_box = [x, y, x + width, y + height]  
                bounding_boxes.append(bounding_box)
    return bounding_boxes

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
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    print(universal_list)
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
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box = []
        print("------------------------")
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = universal_list
        universal_list = []
        boundary_box = []
    print(universal_list)
    return standard_dict_non_mitotic



def extract_bounding_boxes(annotation_file, root_dir):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    annotations = {}
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
            boxes.append([x, y, x + width, y + height])
        annotations[img_path] = boxes
    
    return annotations


transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def tensor_image(image):
    image = T.ToTensor()(image)
    return image

def load_images_and_annotations(root_dir, mitotic_dict, non_mitotic_dict):
    X = []
    y = []
    for filename, mitotic_boxes in mitotic_dict.items():
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to (224, 224)
        img = np.array(img)  # Convert PIL.Image to numpy array
        X.append(img)
        y.append(1)  # Mitotic class label
        
    for filename, non_mitotic_boxes in non_mitotic_dict.items():
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to (224, 224)
        img = np.array(img)  # Convert PIL.Image to numpy array
        X.append(img)
        y.append(0)  # Non-mitotic class label
        
    return np.array(X), np.array(y)

def extract_features(image_path):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = base_model.predict(image)
    return features

def extract_filenames_from_json(json_file, root):
    with open(json_file) as f:
        data = json.load(f)
    
    filename_list = []
    for filename, attributes in data.items():
        filename = filename.replace('.jpeg', '.jpg')
        img_name = attributes['filename'] 
        img_path = os.path.join(root, img_name)  
        file_path  = img_path
        filename_list.append(img_path)
        
    return filename_list

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

# Define root directory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Extract filenames and modify dicts
mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
mitotic_output = print_mitotic(mitotic_annotation_file)
non_mitotic_output = print_filename_bbox(non_mitotic_annotation_file)

modify_dict_inplace(standard_dict_non_mitotic, root)

mitotic_filenames = extract_filenames_from_json(mitotic_annotation_file, root)
mitotic_output = print_mitotic(mitotic_annotation_file)
non_mitotic_output = print_filename_bbox(non_mitotic_annotation_file)

modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



# Define root directory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Load annotations
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)

# Modify dictionaries inplace
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)

# Load images and annotations
X, y = load_images_and_annotations(root, standard_dict_mitotic, standard_dict_non_mitotic)
print(X)
print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

batch_size = 4
steps_per_epoch = len(X_train) // batch_size
epochs = 10

test_img_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"


# Adjusting Dense layer units and activation
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Experiment with different units
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Adjusting learning rate and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Experiment with different learning rates
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

filepath = 'best_model.keras'  # Ensure the filepath ends with .keras

# Define callbacks including ModelCheckpoint with the correct filepath
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
]
# Training the model with callbacks
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
print(history.history.keys())
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
print("CNN is tuned")
print("Starting the code")


bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******\n")

print("****** SelectiVE search ********\n")
proposals = get_proposals(image_path)
print("The proposal regions are:", proposals)
print("Number of Region Proposals:", len(proposals))
print("\n****** SelectiVE search ********\n")

print("****** CNN Layer ********\n")
features = extract_features(image_path)
print(features)
print("\n****** CNN Layer ********\n")

print("****** ROI Pooling Layer ********\n")
output_size =  (7,7)
roi_align_layer = TorchROIAlign(output_size, scaling_factor =0.25)
features =  torch.tensor(features)
proposals = torch.tensor(proposals)
print("Shape of features:", features.shape)
print("Shape of proposals:", proposals.shape)

aligned_features = roi_align_layer(torch.tensor(features), torch.tensor(proposals))
print("\n****** Aligned Features ******\n")
print("Shape of aligned features:", aligned_features.shape)
print(aligned_features)
print("\n****** Aligned Features ******\n")
