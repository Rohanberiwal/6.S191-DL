# YOLOv1 (You Only Look Once)

## Introduction:
- **Authors**: Joseph Redmon et al. (2015)
- **Objective**: Real-time object detection with a single neural network architecture.
- **Innovation**: Introduced the concept of predicting bounding boxes and class probabilities in a single pass through the network.

## Architecture:
- **Grid-based Approach**: Divides the input image into an \( S \times S \) grid.
- **Predictions**: Each grid cell predicts bounding boxes and class probabilities.
- **Network**: Consists of 24 convolutional layers followed by 2 fully connected layers.

## Loss Function:
- **Localization Loss**: Measures the error in predicting bounding box coordinates.
- **Confidence Loss**: Measures the error in predicting objectness score.
- **Combined Loss**: Utilizes a predefined loss function to combine both localization and confidence losses.

## Performance:
- **Real-time Detection**: Achieved real-time object detection with impressive accuracy.
- **Limitations**: Struggled with detecting small objects and suffered from localization errors due to the grid-based approach.

