# Spatial Cross-Entropy Loss

Spatial Cross-Entropy Loss is typically used in tasks that involve pixel-wise classification in images, such as semantic segmentation. In semantic segmentation, each pixel in an image is classified into one of several categories. The spatial cross-entropy loss measures the dissimilarity between the true pixel-wise class labels and the predicted pixel-wise class probabilities output by the model.

## Overview

### Definition
Spatial Cross-Entropy Loss is an extension of the categorical cross-entropy loss to pixel-wise classification tasks. It calculates the cross-entropy loss for each pixel in the image and averages the loss over all pixels.

### Formula
The loss is calculated as:
\[ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) \]
Where:
- \( N \) is the number of pixels in the image,
- \( C \) is the number of classes,
- \( y_{i,c} \) is the true probability of pixel \( i \) belonging to class \( c \),
- \( \hat{y}_{i,c} \) is the predicted probability of pixel \( i \) belonging to class \( c \).

### Usage
This loss function is commonly used in deep learning models for semantic segmentation tasks. Models like Fully Convolutional Networks (FCNs), U-Net, and SegNet, which are designed for pixel-wise classification, often use spatial cross-entropy loss for training.

### Benefits
- **Pixel-wise Accuracy:** Ensures that the model learns to classify each pixel correctly, which is crucial for tasks like semantic segmentation.
- **Differentiable:** Like other cross-entropy loss functions, it is differentiable and can be used with gradient-based optimization algorithms.
