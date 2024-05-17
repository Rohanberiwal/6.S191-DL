# Sigmoid Function in Neural Networks

## Introduction

The Sigmoid function is a popular activation function used in neural networks. It maps any real-valued number into the range (0, 1), which makes it useful for models that need to predict probabilities. The Sigmoid function introduces non-linearity into the model, enabling the network to learn complex patterns.

## Mathematical Definition

The Sigmoid function, also known as the logistic function, is defined as:

\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

where \( e \) is the base of the natural logarithm and \( x \) is the input to the function.

## Properties of the Sigmoid Function

1. **Range**: The output values range between 0 and 1.
2. **Smooth Gradient**: The function has a smooth gradient, which is useful for gradient-based optimization algorithms.
3. **Non-Linearity**: The Sigmoid function introduces non-linearity into the network, allowing it to learn more complex patterns.
4. **Derivative**: The derivative of the Sigmoid function is simple to compute and is given by:
   
   \[ \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) \]

## Role in Neural Networks

### Activation Function

In a neural network, the activation function is used to introduce non-linearity into the model. Without non-linearity, a neural network with multiple layers would behave like a single-layer network, regardless of its depth. This would limit the network's ability to model complex relationships in the data.

### Non-Linearity

Non-linear activation functions like the Sigmoid function enable neural networks to learn and model complex patterns by introducing non-linear transformations at each layer. This non-linearity allows the network to capture intricate relationships between the input features and the target outputs.

### Usage in Logistic Regression

In logistic regression, the Sigmoid function is used to map predicted values to probabilities. The output of the Sigmoid function can be interpreted as the probability of the positive class in a binary classification problem.

## Mathematical Intuition

### Graph of the Sigmoid Function

The graph of the Sigmoid function has an S-shaped curve, which asymptotically approaches 0 as \( x \) approaches negative infinity and 1 as \( x \) approaches positive infinity.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.show()
