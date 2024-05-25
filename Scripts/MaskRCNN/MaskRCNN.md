# Mask Region Convolutional Neural Network

Mask R-CNN is an extension of the Faster R-CNN framework, where a small branch is added to predict object masks in parallel with bounding box detection. The boundary boxes are identified using the Region Proposal Network (RPN) in both Faster R-CNN and Mask R-CNN.

## Overhead of Mask R-CNN over Faster R-CNN

The frame rate per second is lower for Mask R-CNN, typically around 5 fps, compared to Faster R-CNN, which typically achieves frame rates ranging from 5 to 20 fps.

## Mask R-CNN Backbone and Annotation Format

The backbone of Mask R-CNN typically utilizes VGG-16, and the instance segmentation is employed for image segmentation. The annotations are often stored in XML file format.

## Framework Progression: R-CNN to Faster R-CNN

The transition from R-CNN to Faster R-CNN represents a shift in object detection methods from using external region proposal methods to integrating mechanisms for generating region proposals within the model architecture. This transition has led to improvements in both speed and accuracy.

## Output Prediction in Mask R-CNN

In Faster R-CNN, the output consists of boundary box offsets from the Region Proposal Network (RPN) and class label predictions from the classifier (e.g., SVM). In contrast, Mask R-CNN adds a third predictor that outputs binary masks for each object instance. These masks differentiate objects of the same class.

## Backbone Difference in Mask R-CNN

The main difference in the backbone of Mask R-CNN compared to Faster R-CNN is the inclusion of pixel-to-pixel alignment, which is a characteristic of semantic segmentation.

## Overview of Faster R-CNN

Faster R-CNN consists of two stages: 
1. The first stage, known as the Region Proposal Network (RPN), proposes candidate object bounding boxes. 
2. The second stage utilizes RoIPool to extract features from each candidate box, performing classification and bounding box regression.

## Use of ROI Pooling Layer

The ROI pooling layer adaptively resizes irregularly shaped regions of interest (RoIs) into fixed-size feature maps. These maps are then used for classification and bounding box regression.

## Functionality of Mask R-CNN

Mask R-CNN performs the following tasks:
1. Boundary box classification
2. Regression in parallel with the classification
3. Outputting binary masks for each RoI indicating the presence of an object within the boundary box.

## Loss Function in Mask R-CNN

The loss function in Mask R-CNN consists of three components:
1. Classification loss (Lcls), computed using a SoftMax cross-entropy loss function.
2. Boundary box regression loss (Lbox).
3. Mask R-CNN loss (Lmask).

The overall loss is often calculated as the sum of these three losses.
