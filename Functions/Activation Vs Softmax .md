# Activation Functions vs. Softmax in Neural Networks

## Activation Functions

Activation functions are used in each layer of a neural network except the output layer (in most cases). They introduce non-linearity into the model, allowing it to learn complex patterns.

### Common Activation Functions:

1. **ReLU (Rectified Linear Unit)**
   - Formula: \( \text{ReLU}(x) = \max(0, x) \)
   - **Use**: Hidden layers.
   - **Benefits**: Prevents vanishing gradient problem, enables sparse activation.

2. **Sigmoid**
   - Formula: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
   - **Use**: Binary classification problems.
   - **Benefits**: Maps output to a range between 0 and 1.

3. **Tanh (Hyperbolic Tangent)**
   - Formula: \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
   - **Use**: Hidden layers, particularly in recurrent neural networks (RNNs).
   - **Benefits**: Outputs values between -1 and 1.

## Softmax Function

Softmax is used in the output layer of a neural network for multi-class classification problems. It converts raw output scores (logits) into probabilities.

### Formula:
\[ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \]

### Use Cases:
- **Multi-class classification**: Each class is mutually exclusive.
- **Output**: Provides a probability distribution over classes.

## Summary

- **Activation Functions (ReLU, Sigmoid, Tanh)**:
  - Use in hidden layers.
  - Introduce non-linearity.
  - Enable learning complex patterns.

- **Softmax Function**:
  - Use in the output layer for multi-class classification.
  - Converts logits to probabilities.
  - Provides a normalized probability distribution over classes.

## Practical Example

For a neural network designed to classify images of digits (0-9):
- **Hidden Layers**: Use ReLU or Tanh as the activation function.
- **Output Layer**: Use Softmax to produce a probability distribution across the 10 classes.

By using activation functions in the hidden layers and Softmax in the output layer, the network can effectively learn complex patterns and provide meaningful probabilistic predictions.
