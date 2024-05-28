# Difference between Hyperplane and Decision Boundary in SVM

In the context of Support Vector Machines (SVMs), the terms "hyperplane" and "decision boundary" are often used, sometimes interchangeably. However, they have specific meanings and roles. This document clarifies the differences between these two concepts.

## Hyperplane

### Definition

A hyperplane is a flat affine subspace of one dimension less than its ambient space. In an \(n\)-dimensional space, a hyperplane is an \((n-1)\)-dimensional subspace.

### Equation

The equation of a hyperplane in an \(n\)-dimensional space is:

\[ w \cdot x + b = 0 \]

where:
- \( w \) is the weight vector (normal to the hyperplane).
- \( x \) is the feature vector.
- \( b \) is the bias term.

### Characteristics

- **Dimension**: In a 2D space, a hyperplane is a line. In a 3D space, it is a plane. In higher dimensions, it is called a hyperplane.
- **Geometric Entity**: It divides the space into two halves.
- **Perpendicular Vector**: The vector \( w \) is perpendicular to the hyperplane.

## Decision Boundary

### Definition

The decision boundary is a specific hyperplane that SVM uses to separate data points of different classes. It is the hyperplane that SVM finds to achieve the best separation between classes while maximizing the margin.

### Role in SVM

The decision boundary is determined by the SVM algorithm as the optimal hyperplane that separates the two classes with the maximum margin. It is a key concept in classification tasks.

### Characteristics

- **Optimal Separation**: It is the hyperplane that maximizes the margin between the two classes.
- **Margin**: The margin is the distance between the decision boundary and the nearest data points of each class, known as support vectors.
- **Support Vectors**: The data points that lie on the edges of the margin and are closest to the decision boundary.

## Key Differences

- **General vs. Specific**: A hyperplane is a general concept for a flat subspace of one dimension less than its ambient space, whereas the decision boundary is the specific hyperplane chosen by the SVM algorithm to separate classes.
- **Role**: The hyperplane can refer to any separating plane in the feature space, but the decision boundary specifically refers to the hyperplane that optimally separates the classes in SVM.
- **Optimization**: The decision boundary is derived through an optimization process (maximizing the margin), while a hyperplane can be any plane within the space.
- **Context**: In SVM, the decision boundary is the result of the training process, while the term hyperplane is used in a more general geometric sense.

## Conclusion

While the terms hyperplane and decision boundary are related, they are not synonymous. The hyperplane is a general concept of a flat subspace, while the decision boundary is the specific hyperplane identified by the SVM algorithm to optimally separate different classes. Understanding this distinction is important for comprehending how SVMs function in classification tasks.

