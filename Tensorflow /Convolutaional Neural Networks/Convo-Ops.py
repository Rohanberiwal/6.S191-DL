import numpy as np

def convolve(image, kernel, stride=1, padding=0):
    image_padded = np.pad(image, [(padding, padding), (padding, padding)], mode='constant')

    image_h, image_w = image_padded.shape
    kernel_h, kernel_w = kernel.shape
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    output = np.zeros((output_h, output_w))

    for i in range(0, output_h * stride, stride):
        for j in range(0, output_w * stride, stride):
            output[i//stride, j//stride] = np.sum(
                image_padded[i:i+kernel_h, j:j+kernel_w] * kernel
            )
    
    return output

image = np.array([
    [1, 2, 0, 3, 1],
    [4, 6, 1, 2, 1],
    [1, 2, 3, 1, 0],
    [0, 1, 1, 2, 1],
    [2, 4, 1, 2, 3]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

output = convolve(image, kernel, stride=1, padding=1)
print(output)
