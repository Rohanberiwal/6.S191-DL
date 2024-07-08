import numpy as np

def roi_align(feature_maps, rois, output_size):
    # `feature_maps`: 2D numpy array (assumed to be the input feature map)
    # `rois`: List of ROIs, each ROI is a tuple (top-left x, top-left y, bottom-right x, bottom-right y)
    # `output_size`: Desired output size of each ROI (tuple: height, width)

    pooled_maps = []
    for roi in rois:
        # Extract ROI coordinates
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = roi
        roi_height = bottom_right_y - top_left_y
        roi_width = bottom_right_x - top_left_x

        # Calculate sampling points
        y_points = np.linspace(top_left_y, bottom_right_y, output_size[0], endpoint=False)
        x_points = np.linspace(top_left_x, bottom_right_x, output_size[1], endpoint=False)

        # Sample the feature map using bilinear interpolation
        sampled_roi = []
        for y in y_points:
            row = []
            for x in x_points:
                y_low = int(np.floor(y))
                x_low = int(np.floor(x))
                y_high = y_low + 1
                x_high = x_low + 1

                # Bilinear interpolation weights
                ly = y - y_low
                lx = x - x_low
                hy = 1.0 - ly
                hx = 1.0 - lx

                # Ensure interpolation points are within bounds
                y_low = np.clip(y_low, 0, feature_maps.shape[0] - 1)
                y_high = np.clip(y_high, 0, feature_maps.shape[0] - 1)
                x_low = np.clip(x_low, 0, feature_maps.shape[1] - 1)
                x_high = np.clip(x_high, 0, feature_maps.shape[1] - 1)

                # Perform bilinear interpolation
                interpolated_value = (hy * (hx * feature_maps[y_low, x_low] + lx * feature_maps[y_low, x_high]) +
                                      ly * (hx * feature_maps[y_high, x_low] + lx * feature_maps[y_high, x_high]))

                row.append(interpolated_value)
            sampled_roi.append(row)
        
        pooled_maps.append(sampled_roi)

    return np.array(pooled_maps)

# Example usage
FEATURE_MAPS = np.array([[0.3, 0.7, 0.6],
                         [0.8, 0.4, 0.2],
                         [0.1, 0.9, 0.5]])

# Example ROI (top-left x, top-left y, bottom-right x, bottom-right y)
roi_example = (0, 0, 2, 2)
output_size_example = (2, 2)

pooled_maps_example = roi_align(FEATURE_MAPS, [roi_example], output_size_example)
print("Pooled maps:")
print(pooled_maps_example)
