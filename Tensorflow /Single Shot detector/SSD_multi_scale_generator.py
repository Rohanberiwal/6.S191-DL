import numpy as np
import cv2

def generate_multi_scale_maps(feature_map, scales):
    # Initialize an empty list to store multi-scale maps
    multi_scale_maps = []

    # Iterate over the scales
    for scale in scales:
        # Resize the feature map using OpenCV's resize function
        resized_map = cv2.resize(feature_map, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        multi_scale_maps.append(resized_map)

    return multi_scale_maps

# Example usage
feature_map = np.array([[0.3, 0.7, 0.6], [0.8, 0.4, 0.2], [0.1, 0.9, 0.5]])
scales = [0.5, 1.0, 2.0]  # Example scales

multi_scale_maps = generate_multi_scale_maps(feature_map, scales)

# Printing the multi-scale maps
for idx, scaled_map in enumerate(multi_scale_maps):
    print(f"Scale {scales[idx]}:")
    print(scaled_map)
    print()
