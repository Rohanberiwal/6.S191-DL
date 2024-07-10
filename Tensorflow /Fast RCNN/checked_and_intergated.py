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
from tensorflow.keras.applications import ResNet101 
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import torch

import tensorflow as tf
import numpy as np

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

modify_dict_inplace(standard_dict_non_mitotic, root)
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define rootdirectory and annotation files
root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'

# Load annotations
standard_dict_mitotic = print_mitotic(mitotic_annotation_file)
standard_dict_non_mitotic = print_filename_bbox(non_mitotic_annotation_file)

# Modify dictionaries inplace
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
print(standard_dict_non_mitotic)
print(standard_dict_mitotic)
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

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Experiment with different units
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Adjusting learning rate and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Experiment with different learning rates
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

filepath = 'best_model.keras'  
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

epoches  = 100
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy and loss from a Keras history object.

    Parameters:
    - history (History): Keras History object containing training history.
    - save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot as an image file
    plt.close()  # Close the plot to free up memory

# Example usage:
plot_training_history(history, save_path='training_history.png')



image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
print("CNN is tuned")
print("Starting the code")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Force TensorFlow to use GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def remove_empty_feature_maps(features):
    non_empty_indices = []
    for idx, feature_map in enumerate(features):
        if not np.all(feature_map == 0):
            non_empty_indices.append(idx)
    filtered_features = [features[idx] for idx in non_empty_indices]
    return filtered_features, non_empty_indices

print("This are the globaal dicts")
print(standard_dict_mitotic)
print("\n")
print(standard_dict_non_mitotic)
print("\n")
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
print("\n****** CNN Layer ********\n")
print("This is the code  for the ROI pooling network")

print("tHIS IS THE TESTIFN SEGEMNET")
for i in features  :
    print(np.array(i))

print("This is the code for the matric pooling")

import torch
import torch.nn.functional as F

def crop_and_resize_torch(image_tensor, bbox, target_size):
    """
    Crop and resize a region of interest (ROI) from an image tensor (PyTorch).

    Parameters:
    - image_tensor (torch.Tensor): The input image tensor.
    - bbox (list): List containing bounding box coordinates [xmin, ymin, xmax, ymax].
    - target_size (tuple): Target size (height, width) for resizing.

    Returns:
    - resized_roi (torch.Tensor): Resized ROI as a PyTorch tensor.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Ensure integer values
    roi = image_tensor[:, :, ymin:ymax, xmin:xmax]  # Ensure correct channel dimension
    resized_roi = F.interpolate(roi, size=target_size, mode='nearest')
    return resized_roi.squeeze(0)



def convert_to_normalized_boxes(proposals, image_height, image_width):
    """
    Convert bounding box proposals to normalized coordinates.

    Parameters:
    - proposals (numpy.ndarray): Array of bounding box proposals in format [y_min, x_min, height, width].
    - image_height (int): Height of the input image or feature map.
    - image_width (int): Width of the input image or feature map.

    Returns:
    - normalized_boxes (numpy.ndarray): Array of normalized bounding boxes in format [y_min, x_min, y_max, x_max].
    """
    normalized_boxes = np.zeros_like(proposals, dtype=np.float32)

    for i, proposal in enumerate(proposals):
        y_min, x_min, height, width = proposal
        y_max = y_min + height
        x_max = x_min + width

        # Normalize coordinates
        normalized_boxes[i, 0] = y_min / image_height
        normalized_boxes[i, 1] = x_min / image_width
        normalized_boxes[i, 2] = y_max / image_height
        normalized_boxes[i, 3] = x_max / image_width

    return normalized_boxes

def roi_align(feature_maps, boxes, output_height, output_width):
    """
    `feature_maps` is a list of 2-D arrays, each representing an input feature map
    `boxes` is a list of lists, where each inner list represents a bounding box [x, y, w, h]
    `output_height` and `output_width` are the desired spatial size of output feature map
    """
    num_feature_maps = len(feature_maps)
    num_boxes = len(boxes)
    output_feature_maps = []

    for fmap_idx in range(num_feature_maps):
        image = feature_maps[fmap_idx]
        img_height, img_width = image.shape
        fmap_boxes = boxes[fmap_idx]
        fmap_feature_maps = []

        for box in fmap_boxes:
            x, y, w, h = box
            feature_map = []

            for i in range(output_height):
                for j in range(output_width):
                    # Calculate coordinates in the original image space
                    y_orig = y + i * (h / output_height)
                    x_orig = x + j * (w / output_width)

                    y_l = int(np.floor(y_orig))
                    y_h = int(np.ceil(y_orig))
                    x_l = int(np.floor(x_orig))
                    x_h = int(np.ceil(x_orig))

                    # Clip indices to stay within image bounds
                    y_l = np.clip(y_l, 0, img_height - 1)
                    y_h = np.clip(y_h, 0, img_height - 1)
                    x_l = np.clip(x_l, 0, img_width - 1)
                    x_h = np.clip(x_h, 0, img_width - 1)

                    a = image[y_l, x_l]
                    b = image[y_l, x_h]
                    c = image[y_h, x_l]
                    d = image[y_h, x_h]

                    y_weight = y_orig - y_l
                    x_weight = x_orig - x_l

                    val = a * (1 - x_weight) * (1 - y_weight) + \
                          b * x_weight * (1 - y_weight) + \
                          c * y_weight * (1 - x_weight) + \
                          d * x_weight * y_weight

                    feature_map.append(val)

            fmap_feature_maps.append(np.array(feature_map).reshape(output_height, output_width))

        output_feature_maps.append(fmap_feature_maps)

    return output_feature_maps




from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import numpy as np

avg_pool = GlobalAveragePooling2D()(base_model.output)
max_pool = GlobalMaxPooling2D()(base_model.output)

# Create feature extraction model for pooled outputs
avg_pool_model = Model(inputs=base_model.input, outputs=avg_pool)
max_pool_model = Model(inputs=base_model.input, outputs=max_pool)


# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Generate pooled feature maps
avg_pool_map = avg_pool_model.predict(img_array)
max_pool_map = max_pool_model.predict(img_array)

# Print the shapes of pooled feature maps
print("Average Pooling Feature map shape:", avg_pool_map.shape)
print("Max Pooling Feature map shape:", max_pool_map.shape)

print(avg_pool_map)
print(max_pool_map)
