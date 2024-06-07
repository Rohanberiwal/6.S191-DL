import numpy as np

feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

def roi_pooling(feature_map, pooled_size):
    h, w = feature_map.shape
    ph, pw = pooled_size
    bin_size_h = h // ph
    bin_size_w = w // pw

    pooled_feature_map = np.zeros((ph, pw))
    spatial_bins = []

    for i in range(ph):
        row_bins = []
        for j in range(pw):
            h_start = i * bin_size_h
            h_end = (i + 1) * bin_size_h
            w_start = j * bin_size_w
            w_end = (j + 1) * bin_size_w

            bin_region = feature_map[h_start:h_end, w_start:w_end]
            pooled_feature_map[i, j] = np.max(bin_region)

            row_bins.append(bin_region)

        spatial_bins.append(row_bins)

    return pooled_feature_map, spatial_bins

pooled_size = (2, 2)
pooled_feature_map, spatial_bins = roi_pooling(feature_map, pooled_size)

feature_vector = pooled_feature_map.flatten()



print("Pooled Feature Map (2x2):")
print(pooled_feature_map)

print("n")
print("This is the inital feature map and below are the spatial bins for the same ")
print(feature_map)


print("\nSpatial Bins:")
for i, row in enumerate(spatial_bins):
    for j, bin_region in enumerate(row):
        print(f"Bin ({i}, {j}):")
        print(bin_region)

print("\nFlattened Feature Vector:")
print(feature_vector)
