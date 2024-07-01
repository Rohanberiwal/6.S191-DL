import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn import svm
from tensorflow.keras.layers import Conv2D
from numba import cuda

import os

class FCN(tf.keras.Model):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv3 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv4 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv5 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv6 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv7 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv8 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv9 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv10 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv11 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv12 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv13 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv14 = Conv2D(4096, 7, activation='relu', padding='same')
        self.conv15 = Conv2D(4096, 1, activation='relu', padding='same')
        self.conv16 = Conv2D(num_classes, 1, activation='softmax', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        return x

print(tf.__version__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("check 1 complete")
vgg_model = VGG16(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)
print("check 2 complete")

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = cv2.imread(image_path)

@cuda.jit
def hsv_mask_kernel(image, mask, lower_yellow, upper_yellow):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        r, g, b = image[x, y, 2], image[x, y, 1], image[x, y, 0]
        v = max(r, g, b)
        s = 0 if v == 0 else (v - min(r, g, b)) / v
        h = 0
        if s != 0:
            if v == r:
                h = (60 * (g - b) / (v - min(r, g, b))) % 360
            elif v == g:
                h = (60 * (b - r) / (v - min(r, g, b))) + 120
            elif v == b:
                h = (60 * (r - g) / (v - min(r, g, b))) + 240
        
        if (lower_yellow[0] <= h <= upper_yellow[0] and 
            lower_yellow[1] <= s <= upper_yellow[1] and 
            lower_yellow[2] <= v <= upper_yellow[2]):
            mask[x, y] = 255
        else:
            mask[x, y] = 0

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
yellow_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(image.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(image.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

hsv_mask_kernel[blockspergrid, threadsperblock](image, yellow_mask, lower_yellow, upper_yellow)
print("check 3 complete")

contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_bounding_rects = [cv2.boundingRect(contour) for contour in contours]
print("Number of yellow bounding rectangles found:", len(yellow_bounding_rects))
print("check 4 complete")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
print("check 5 complete")
ss.switchToSelectiveSearchFast()
rects = ss.process()
print("Number of rectangles found by selective search:", len(rects))
print("check 6 complete")

filtered_rects = []
for (x, y, w, h) in rects:
    for (yellow_x, yellow_y, yellow_w, yellow_h) in yellow_bounding_rects:
        if x >= yellow_x and y >= yellow_y and x + w <= yellow_x + yellow_w and y + h <= yellow_y + yellow_h:
            filtered_rects.append((x, y, w, h))
            break

print("Number of filtered rectangles:", len(filtered_rects))
print("check 7 complete")

roi_images = []
for (x, y, w, h) in filtered_rects:
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (224, 224))  
    roi = np.expand_dims(roi, axis=0) 
    roi_images.append(roi)

if not roi_images:
    print("No regions of interest found. Exiting.")
else:
    print("check 8 complete")

    roi_images = np.concatenate(roi_images, axis=0)
    roi_images = preprocess_input(roi_images)
    features = feature_extractor.predict(roi_images)
    print("check 9 complete")

    print("Last check left ")
    for i, feature_map in enumerate(features):
        print(f"Feature maps for bounding box {i+1}:")
        print(feature_map)

    print("All checks done till CNN")

    num_classes = 2  
    model = FCN(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = np.array([1, 0, 1, 0]) 

    fcn_output = model.predict(features)
    fcn_output_flattened = fcn_output.reshape(fcn_output.shape[0], -1)

    print("SVM on ")
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(fcn_output_flattened, y_train)
    print("predictions")
    predictions = svm_classifier.predict(fcn_output_flattened)
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            print(f"Region {i+1}: Mitotic")
        else:
            print(f"Region {i+1}: Non-Mitotic")









