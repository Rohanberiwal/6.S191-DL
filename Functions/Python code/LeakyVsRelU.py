import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Generate input values using np.linspace
x_values = np.linspace(-10, 10, 100)  # Generate 100 values from -10 to 10

# Calculate corresponding output values using ReLU and Leaky ReLU functions
y_relu = relu(x_values)
y_leaky_relu = leaky_relu(x_values)

# Plot both ReLU and Leaky ReLU functions
plt.plot(x_values, y_relu, label='ReLU', color='blue')
plt.plot(x_values, y_leaky_relu, label='Leaky ReLU', color='red')
plt.title('ReLU vs Leaky ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
