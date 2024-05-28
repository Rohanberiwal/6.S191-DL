# Kernel Types in SVM

Support Vector Machines (SVMs) are powerful classification algorithms that can handle non-linear data by using kernel functions. Kernels implicitly map the input features into higher-dimensional spaces, enabling SVMs to find non-linear decision boundaries. Here, we discuss the three common types of kernels: linear, polynomial, and radial basis function (RBF) kernels.

## 1. Linear Kernel

### Definition

The linear kernel is the simplest kernel function. It does not perform any transformation and works directly with the input features.

### Formula

\[ K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j \]

### Characteristics

- **Simplicity**: The linear kernel is computationally efficient and straightforward.
- **Use Case**: Suitable for linearly separable data or when the number of features is very high relative to the number of samples.
- **Decision Boundary**: A linear hyperplane.

### Example

If \(\mathbf{x}_i = [x_{i1}, x_{i2}]\) and \(\mathbf{x}_j = [x_{j1}, x_{j2}]\), then:

\[ K(\mathbf{x}_i, \mathbf{x}_j) = x_{i1} x_{j1} + x_{i2} x_{j2} \]

## 2. Polynomial Kernel

### Definition

The polynomial kernel represents the similarity of vectors in a feature space over polynomials of the original variables.

### Formula

\[ K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d \]

where:
- \( d \) is the degree of the polynomial.
- \( c \) is a constant term (trade-off parameter).

### Characteristics

- **Flexibility**: Capable of handling complex data by adjusting the degree \( d \).
- **Non-linear Decision Boundaries**: Can model non-linear relationships.
- **Parameter Sensitivity**: The performance depends on the choice of \( d \) and \( c \).

### Example

For \( d = 2 \) and \( c = 1 \):

\[ K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + 1)^2 \]

## 3. Radial Basis Function (RBF) Kernel

### Definition

The RBF kernel, also known as the Gaussian kernel, maps the input features into an infinite-dimensional space.

### Formula

\[ K(\mathbf{x}_i, \mathbf{x}_j) = \exp \left( -\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2} \right) \]

where:
- \(\sigma\) is a parameter that defines the width of the Gaussian function.

### Characteristics

- **Versatility**: Suitable for non-linear data and can handle a variety of patterns.
- **Localized Influence**: The influence of a single training example decreases with distance.
- **Parameter Sensitivity**: Requires careful tuning of the \(\sigma\) parameter.

### Example

If \(\mathbf{x}_i = [x_{i1}, x_{i2}]\) and \(\mathbf{x}_j = [x_{j1}, x_{j2}]\), then:

\[ K(\mathbf{x}_i, \mathbf{x}_j) = \exp \left( -\frac{(x_{i1} - x_{j1})^2 + (x_{i2} - x_{j2})^2}{2\sigma^2} \right) \]

## Choosing the Right Kernel

- **Linear Kernel**: Use when the data is linearly separable or when dealing with a high-dimensional feature space.
- **Polynomial Kernel**: Use for data with polynomial relationships; adjust the degree \( d \) to control the flexibility.
- **RBF Kernel**: Use for most non-linear datasets; it is a versatile default choice but requires careful parameter tuning.

## Conclusion

Kernels enable SVMs to handle non-linear data by transforming the input space into higher dimensions. The choice of kernel—linear, polynomial, or RBF—depends on the nature of the data and the specific problem at hand. Understanding the properties and appropriate use cases of each kernel helps in effectively applying SVMs to various classification tasks.

