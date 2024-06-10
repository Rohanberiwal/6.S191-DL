
"""
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
## Feature extracted 
print("check 2 complete")

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100]) ## RGB 
upper_yellow = np.array([30, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  ## use to yellopw edge for thr mitosis 
print("check 3 complete")

contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours) ## check countours 
yellow_bounding_rects = [cv2.boundingRect(contour) for contour in contours]
print("check 4 complete")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() ## 2000 image 
ss.setBaseImage(image)
print("check 5 complete")
ss.switchToSelectiveSearchFast()
rects = ss.process()
print(rects)
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
    roi = cv2.resize(roi, (224, 224))  ## warped 
    roi = np.expand_dims(roi, axis=0) 
    roi_images.append(roi)
    
print("check 8 complete")

roi_images = np.concatenate(roi_images, axis=0)
roi_images = preprocess_input(roi_images) 
features = feature_extractor.predict(roi_images) ## roi image extractor 
print("check 9 complete")

print("Last check left ")
for i, feature_map in enumerate(features):
    print(f"Feature maps for bounding box {i+1}:")
    print(feature_map)
    
print("All checks done till CNN")
num_classes = 2  
model = FCN(num_classes) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y_train = np.array([1, 0, 1, 0])  # hard 

fcn_output = model.predict(features)
print("fcn outputs ->> feature vectors")
print(fcn_output)

##Issue starts here 
print("SVM on ")
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(fcn_output, y_train)
print("predictions")
predictions = svm_classifier.predict(fcn_output)
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Region {i+1}: Mitotic")
    else:
        print(f"Region {i+1}: Non-Mitotic")

"""
