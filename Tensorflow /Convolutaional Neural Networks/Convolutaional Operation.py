import numpy as np

# Input image
image = np.array([
    [1, 2, 0, 3, 1],
    [4, 6, 1, 2, 1],
    [1, 2, 3, 1, 0],
    [0, 1, 1, 2, 1],
    [2, 4, 1, 2, 3]
])

# Filte
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Convolution operation
def convolve(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[i, j] = np.sum(image[i:i+kernel_h, j:j+kernel_w] * kernel)
    
    return output

output = convolve(image, kernel)
print(output)
