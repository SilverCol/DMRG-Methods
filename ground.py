import numpy as np
from numpy import linalg as la

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
    # Odd spins
    for j in range(1, n + 1, 2):
        # Create the set of 4 matrices 
        C00 = U0 * la.multi_dot(L[j-1], B[j][0], L[j], B[j][0], L[j+1])

        t01 = la.multi_dot(B[j][0], L[j], B[j][1])
        t10 = la.multi_dot(B[j][1], L[j], B[j][0])
        C01 = la.multi_dot(L[j-1], U1 * t01 + U2 * t10, L[j+1])
        C10 = la.multi_dot(L[j-1], U1 * t10 + U2 * t01, L[j+1])

        C11 = U0 * la.multi_dot(L[j-1], B[j][1], L[j], B[j][1], L[j+1])

        # Assemble the Q matrix
        pass

    # Even spins
    for j in range(2, n + 1, 2):
        pass
