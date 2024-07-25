
def resize_or_pad(arr, target_shape):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input 'arr' must be a NumPy array.")
    
    if len(target_shape) != 2:
        raise ValueError("Target shape must be a tuple of (height, width).")
    
    print(f"Original array shape: {arr.shape}")
    print(f"Target shape: {target_shape}")

    if arr.size == 0:
        raise ValueError("Input array is empty.")
    
    if len(arr.shape) == 2:
        # 2D array
        resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    elif len(arr.shape) == 3:
        # 3D array (height, width, channels)
        height, width, channels = arr.shape
        resized = np.zeros((target_shape[0], target_shape[1], channels), dtype=arr.dtype)
        for c in range(channels):
            resized[:, :, c] = cv2.resize(arr[:, :, c], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Array must be 2D or 3D for resizing.")
    
    return resized

class ROIPoolingLayer(nn.Module):
    def __init__(self, output_size):
        super(ROIPoolingLayer, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois):
        x1 = rois[:, 0]  # x
        y1 = rois[:, 1]  # y
        x2 = rois[:, 0] + rois[:, 2]  # x + w
        y2 = rois[:, 1] + rois[:, 3]  # y + h
        rois = torch.stack([x1, y1, x2, y2], dim=1)
        rois = torch.cat([torch.zeros(rois.size(0), 1), rois], dim=1)  # Shape [R, 5]
        return roi_pool(feature_map, rois, output_size=self.output_size)

def process_feature_maps_and_rois(feature_maps, rois_list, target_shape):
    roipool = ROIPoolingLayer(output_size=target_shape)
    pooled_features_list = []

    for feature_map, rois in zip(feature_maps, rois_list):
        try:
            # Resize each feature map slice individually
            feature_map_resized = np.array([resize_or_pad(feature_map[:, :, c], target_shape) for c in range(feature_map.shape[2])])
            feature_map_resized = np.transpose(feature_map_resized, (1, 2, 0))  # Reshape to (height, width, channels)
            feature_map_tensor = torch.tensor(feature_map_resized, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Convert ROIs to correct format
            if rois.shape[0] == 4 and rois.shape[1] == 1:
                rois = rois.T  # Transpose to shape [1, 4]
            elif rois.shape[0] != 4:
                raise ValueError("ROIs must have shape [4, 1] or [N, 4].")
            
            rois_tensor = torch.tensor(rois, dtype=torch.float32)
            
            print("Feature Map (Resized):", feature_map_tensor.shape)
            print("ROIs (Resized):", rois_tensor.shape)

            # Perform ROI pooling
            pooled_features = roipool(feature_map_tensor, rois_tensor)
            pooled_features_list.append(pooled_features)
            print("Pooled Features Shape:", pooled_features.shape)
            print("Pooled Features:", pooled_features)
        except Exception as e:
            print(f"Error processing feature map and ROIs: {e}")

    print("Half pipeline is over")
    return pooled_features_list

