import tensorflow as tf
from tensorflow.keras.layers import Layer

class RoiAlign(Layer):
    def __init__(self, pool_size, **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        feature_map, rois = inputs

        # Unpack ROI parameters (assuming rois are in [x, y, w, h] format)
        roi_batch_indices = tf.cast(rois[:, 0], dtype=tf.int32)
        rois = rois[:, 1:]  # Exclude the batch index

        # Calculate ROI dimensions
        roi_width = rois[:, 2]
        roi_height = rois[:, 3]

        # Calculate ROI bin size
        bin_width = roi_width / self.pool_size[0]
        bin_height = roi_height / self.pool_size[1]

        # Initialize list to store pooled features
        pooled_features = []

        # Iterate over each ROI
        for roi_idx in range(tf.shape(rois)[0]):
            # Extract ROI
            roi = rois[roi_idx]
            bin_width_i = bin_width[roi_idx]
            bin_height_i = bin_height[roi_idx]

            # Calculate x1, y1, x2, y2 from [x, y, w, h]
            x1 = roi[0]
            y1 = roi[1]
            x2 = x1 + roi_width[roi_idx]
            y2 = y1 + roi_height[roi_idx]

            # Generate grid of bin coordinates
            grid_x = tf.linspace(x1, x2, self.pool_size[0] + 1)
            grid_y = tf.linspace(y1, y2, self.pool_size[1] + 1)

            # Iterate over each bin in the grid
            roi_features = []
            for y in range(self.pool_size[1]):
                for x in range(self.pool_size[0]):
                    # Define bin boundaries
                    x_left = tf.cast(tf.math.floor(grid_x[x]), dtype=tf.int32)
                    x_right = tf.cast(tf.math.ceil(grid_x[x + 1]), dtype=tf.int32)
                    y_top = tf.cast(tf.math.floor(grid_y[y]), dtype=tf.int32)
                    y_bottom = tf.cast(tf.math.ceil(grid_y[y + 1]), dtype=tf.int32)

                    # Extract bin from feature map
                    bin_features = feature_map[roi_batch_indices[roi_idx], y_top:y_bottom, x_left:x_right]

                    # Perform bilinear interpolation
                    x_l = grid_x[x] - tf.cast(x_left, dtype=tf.float32)
                    x_r = tf.cast(x_right, dtype=tf.float32) - grid_x[x]
                    y_t = grid_y[y] - tf.cast(y_top, dtype=tf.float32)
                    y_b = tf.cast(y_bottom, dtype=tf.float32) - grid_y[y]

                    interpolated_feature = (
                        bin_features[0, 0] * x_r * y_b +
                        bin_features[0, 1] * x_l * y_b +
                        bin_features[1, 0] * x_r * y_t +
                        bin_features[1, 1] * x_l * y_t
                    ) / ((x_l + x_r) * (y_t + y_b))

                    roi_features.append(interpolated_feature)

            # Append pooled features for current ROI
            pooled_features.append(tf.reshape(tf.stack(roi_features), (self.pool_size[1], self.pool_size[0])))

        # Stack pooled features into final output tensor
        pooled_features = tf.stack(pooled_features)

        # Print pooled feature map (for debugging or demonstration purposes)
        print("Pooled Feature Map:")
        print(pooled_features)

        return pooled_features

    def compute_output_shape(self, input_shape):
        # Assuming feature_map shape is (batch_size, height, width, channels)
        return (input_shape[0][0], self.pool_size[1], self.pool_size[0])




pool_size = (7, 7)  # Example pool size, adjust as needed
roi_align_layer = RoiAlign(pool_size)

# Initialize list to store pooled feature maps
pooled_feature_maps = []

# Iterate over each feature map and corresponding RoI
for feature_map, roi in zip(featured_map,rois):
    # Reshape feature_map if needed to include batch dimension
    feature_map_with_batch = tf.expand_dims(feature_map, axis=0)
    
    # Call RoiAlign layer with current feature map and RoI
    pooled_features = roi_align_layer([feature_map_with_batch, tf.expand_dims(roi, axis=0)])
    
    # Append pooled feature map to list
    pooled_feature_maps.append(pooled_features.numpy())

print("end of the Align  layer ")
