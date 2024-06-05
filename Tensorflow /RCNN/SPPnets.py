import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
import os

print(tf.__version__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SpatialPyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):
        self.pool_list = pool_list
        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

    def call(self, inputs):
        outputs = []
        for pool_size in self.pool_list:
            pool = tf.image.resize(inputs, [pool_size, pool_size])
            pool = tf.keras.layers.MaxPooling2D(pool_size)(pool)
            outputs.append(tf.reshape(pool, (inputs.shape[0], -1)))
        return tf.concat(outputs, axis=-1)

print("check 1 complete")
vgg_model = VGG16(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)
print("check 2 complete")

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
print("check 3 complete")

contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_bounding_rects = [cv2.boundingRect(contour) for contour in contours]
print("check 4 complete")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
print("check 5 complete")
ss.switchToSelectiveSearchFast()
rects = ss.process()
print("check 6 complete")

filtered_rects = []
for (x, y, w, h) in rects:
    for (yellow_x, yellow_y, yellow_w, yellow_h) in yellow_bounding_rects:
        if x >= yellow_x and y >= yellow_y and x + w <= yellow_x + yellow_w and y + h <= yellow_y + yellow_h:
            filtered_rects.append((x, y, w, h))
            break
        
print("check 7 complete")

roi_images = []
for (x, y, w, h) in filtered_rects:
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (224, 224))  
    roi = np.expand_dims(roi, axis=0) 
    roi_images.append(roi)
    
print("check 8 complete")

if len(roi_images) > 0:
    roi_images = np.concatenate(roi_images, axis=0)
    roi_images = preprocess_input(roi_images)
    features = feature_extractor.predict(roi_images)
    print("check 9 complete")
    pool_list = [1, 2, 4]
    spp_layer = SpatialPyramidPooling(pool_list)
    spp_features = spp_layer(features)
    
    print("SPP output:")
    print(spp_features.numpy())

    for (x, y, w, h) in filtered_rects:
        print("Code still running")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("Last check left")
    for i, feature_map in enumerate(spp_features):
        print(f"SPP features for bounding box {i+1}:")
        print(feature_map)
    
    print("All checks done till CNN")
else:
    print("No ROI images found to process.")
