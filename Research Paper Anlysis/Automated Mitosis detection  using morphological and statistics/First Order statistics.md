
## First-Order Statistical Features

In the BR image, we extract five first-order statistical features including mean, median, variance, kurtosis, and skewness of each segmented candidate. Here's a description of each feature:

1. **Mean**:
   - The mean represents the average intensity value of the pixels within the segmented candidate region. It is calculated by summing up all pixel values within the region and dividing by the total number of pixels.

2. **Median**:
   - The median is the middle value of the sorted pixel intensity values within the segmented candidate region. It is less sensitive to extreme pixel values compared to the mean and provides a measure of central tendency.

3. **Variance**:
   - Variance measures the dispersion or spread of pixel intensity values within the segmented candidate region. It quantifies how much the pixel values deviate from the mean value. Higher variance indicates greater variability in intensity values.

4. **Kurtosis**:
   - Kurtosis measures the peakedness or flatness of the distribution of pixel intensity values within the segmented candidate region. Positive kurtosis indicates a relatively peaked distribution, while negative kurtosis indicates a relatively flat distribution.

5. **Skewness**:
   - Skewness measures the asymmetry of the distribution of pixel intensity values within the segmented candidate region. Positive skewness indicates a longer tail on the right side of the distribution, while negative skewness indicates a longer tail on the left side.

These statistical features provide valuable information about the intensity characteristics and distribution of pixel values within segmented regions, which can be used for various purposes such as image classification, segmentation evaluation, and feature-based analysis.
