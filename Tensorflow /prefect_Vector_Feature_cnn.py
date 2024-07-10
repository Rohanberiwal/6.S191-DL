from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
x = base_model.output
x = GlobalAveragePooling2D()(x)

feature_model = Model(inputs=base_model.input, outputs=x)
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Generate feature vector
features = feature_model.predict(img_array)
print("Feature vector shape:", features.shape)
print("Feature vector:")
print(features)

print()
