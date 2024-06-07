import numpy as np

def roi_pooling_forward(feature_map, boxes, output_size):
    pooled_features = []
    for box in boxes:
        y_min, x_min, y_max, x_max = box
        roi_height = y_max - y_min
        roi_width = x_max - x_min
        bin_height = roi_height / output_size[0]
        bin_width = roi_width / output_size[1]
        pooled_feature = np.zeros(output_size)
        for y in range(output_size[0]):
            for x in range(output_size[1]):
                start_y = int(np.floor(y * bin_height)) + y_min
                start_x = int(np.floor(x * bin_width)) + x_min
                end_y = int(np.ceil((y + 1) * bin_height)) + y_min
                end_x = int(np.ceil((x + 1) * bin_width)) + x_min
                pooled_feature[y, x] = np.max(feature_map[start_y:end_y, start_x:end_x])

        pooled_features.append(pooled_feature.flatten())

    return pooled_features
feature_map = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])

boxes = [[0, 0, 2, 2], [1, 1, 3, 3]]
output_size = (2, 2)

pooled_features_forward = roi_pooling_forward(feature_map, boxes, output_size)

print("Feature vector after the forward pass:")
for i, feature in enumerate(pooled_features_forward):
    print("Box", i+1, ":", feature)


def roi_pooling_backward(pooled_features, gradient):
    pooled_gradients = [gradient[i].reshape(output_size) for i in range(len(gradient))]
    gradient_map = np.zeros_like(feature_map)
    for i, box in enumerate(boxes):
        y_min, x_min, y_max, x_max = box
        roi_height = y_max - y_min
        roi_width = x_max - x_min
        bin_height = roi_height / output_size[0]
        bin_width = roi_width / output_size[1]

        pooled_gradient = pooled_gradients[i]

        for y in range(output_size[0]):
            for x in range(output_size[1]):
                start_y = int(np.floor(y * bin_height)) + y_min
                start_x = int(np.floor(x * bin_width)) + x_min
                end_y = int(np.ceil((y + 1) * bin_height)) + y_min
                end_x = int(np.ceil((x + 1) * bin_width)) + x_min
                max_idx = np.argmax(feature_map[start_y:end_y, start_x:end_x])
                max_idx_y, max_idx_x = np.unravel_index(max_idx, (end_y - start_y, end_x - start_x))
                gradient_map[start_y + max_idx_y, start_x + max_idx_x] += pooled_gradient[y, x]

    return gradient_map
gradient = [np.ones(output_size) for _ in range(len(pooled_features_forward))]

gradient_map_backward = roi_pooling_backward(pooled_features_forward, gradient)

print("\nGradient map after the backward pass:")
print(gradient_map_backward)
