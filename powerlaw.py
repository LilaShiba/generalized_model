import matplotlib.pyplot as plt
import numpy as np

# Define the constant and power law exponent
a = 1
b = 2

# Define the range of values for the independent variable
x = np.linspace(1, 100, 10000)

# Calculate the dependent variable using the power law equation
y = a * x**b

# Plot the power law relationship
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power Law Relationship (y = ax^b)')
plt.show()