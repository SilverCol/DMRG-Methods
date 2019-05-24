import numpy as np

# Create a Neel initial MPA
n = 100
B = [ [1 - j%2, j%2] for j in range(n + 2)]
L = [1 for j in range(n + 2)]

# Set propagation constants
steps = 1000
z = -.01
U0 = np.exp(z)
U1 = np.exp(-z) * np.cosh(2*z) 
U2 = np.exp(-z) * np.sinh(2*z)

# Propagate
for step in range(steps):
    pass
