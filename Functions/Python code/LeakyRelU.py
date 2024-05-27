import numpy as np
import matplotlib.pyplot as plt

# Define the Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Generate input values using np.linspace
x_values = np.linspace(-10, 10, 100)  # Generate 100 values from -10 to 10

# Calculate corresponding output values using Leaky ReLU function
alpha = 0.01  # Define the leakage coefficient
y_values = leaky_relu(x_values, alpha)

# Plot the Leaky ReLU function
plt.plot(x_values, y_values, label='Leaky ReLU', color='red')
plt.title('Leaky ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
