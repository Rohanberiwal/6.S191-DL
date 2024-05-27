import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x_values = np.linspace(-10, 10, 100)  
y_values = relu(x_values)

plt.plot(x_values, y_values, label='ReLU', color='blue')
plt.title('ReLU Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()


