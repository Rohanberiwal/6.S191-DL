## Laplacian of Gaussian (LoG) Filter

The Laplacian of Gaussian (LoG) filter is an edge-detection technique that combines Gaussian smoothing with the Laplacian operator to highlight regions with rapid intensity changes (edges) in an image. It is particularly useful for detecting edges and fine details, such as the boundaries of nuclei in histological images.

### How the LoG Filter Works

1. **Gaussian Smoothing**:
   - Smooth the image using a Gaussian filter to reduce noise and minor variations. The amount of smoothing is controlled by the standard deviation \( \sigma \).

2. **Laplacian Operator**:
   - Apply the Laplacian operator to the smoothed image. This operator calculates the second derivative of the image intensity, emphasizing regions where the intensity changes rapidly.

3. **Combined Operation**:
   - The LoG filter performs these steps in a single convolution operation with a kernel that approximates the Laplacian of the Gaussian function.

## Laplacian of Gaussian (LoG) Filter

The formula for the LoG filter is:
\[ \text{LoG}(x, y) = -\frac{1}{\pi \sigma^4} \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) e^{-\frac{x^2 + y^2}{2\sigma^2}} \]

- \( x \) and \( y \) are the coordinates of a pixel in the image.
- \( \sigma \) is the standard deviation of the Gaussian distribution, determining the level of smoothing.
- The term inside the parenthesis, \( \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) \), represents the Laplacian component.
- The exponential term, \( e^{-\frac{x^2 + y^2}{2\sigma^2}} \), represents the Gaussian smoothing component.

### Summary

- The **LoG filter** is used to detect edges in an image by highlighting areas with rapid intensity changes.
- It **smooths the image** first to reduce noise and then applies the **Laplacian operator** to find edges.
- The **mathematical formula** combines both operations into one, emphasizing the edges while smoothing the image.

