
def roi_align(feature_maps, boxes, output_height, output_width):
    """
    `feature_maps` is a list of 2-D arrays, each representing an input feature map
    `boxes` is a list of lists, where each inner list represents a bounding box [x, y, w, h]
    `output_height` and `output_width` are the desired spatial size of output feature map
    """
    num_feature_maps = len(feature_maps)
    num_boxes = len(boxes)
    output_feature_maps = []

    for fmap_idx in range(num_feature_maps):
        image = feature_maps[fmap_idx]
        img_height, img_width = image.shape
        fmap_boxes = boxes[fmap_idx]
        fmap_feature_maps = []

        for box in fmap_boxes:
            x, y, w, h = box
            feature_map = []

            for i in range(output_height):
                for j in range(output_width):
                    # Calculate coordinates in the original image space
                    y_orig = y + i * (h / output_height)
                    x_orig = x + j * (w / output_width)

                    y_l = int(np.floor(y_orig))
                    y_h = int(np.ceil(y_orig))
                    x_l = int(np.floor(x_orig))
                    x_h = int(np.ceil(x_orig))

                    # Clip indices to stay within image bounds
                    y_l = np.clip(y_l, 0, img_height - 1)
                    y_h = np.clip(y_h, 0, img_height - 1)
                    x_l = np.clip(x_l, 0, img_width - 1)
                    x_h = np.clip(x_h, 0, img_width - 1)

                    a = image[y_l, x_l]
                    b = image[y_l, x_h]
                    c = image[y_h, x_l]
                    d = image[y_h, x_h]

                    y_weight = y_orig - y_l
                    x_weight = x_orig - x_l

                    val = a * (1 - x_weight) * (1 - y_weight) + \
                          b * x_weight * (1 - y_weight) + \
                          c * y_weight * (1 - x_weight) + \
                          d * x_weight * y_weight

                    feature_map.append(val)

            fmap_feature_maps.append(np.array(feature_map).reshape(output_height, output_width))

        output_feature_maps.append(fmap_feature_maps)

    return output_feature_maps




output_feature_map = roi_align(features, box, output_height =14 , output_width =14)
for fmap_idx, fmap_feature_maps in enumerate(output_feature_map):
    print(f"\nFeature map {fmap_idx+1}:")
    for box_idx, feature_map in enumerate(fmap_feature_maps):
        print("  Output feature map:")
        print(feature_map)
