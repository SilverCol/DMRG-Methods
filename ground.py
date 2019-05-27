import numpy as np
from numpy import linalg as la

# Create a Neel initial MPA
n = 20
B = [ [np.array([[1 - j%2]]), np.array([[j%2]])] for j in range(n + 2)]
L = [np.array([1]) for j in range(n + 2)]

# Set propagation constants
steps = 1000
z = -.01
error = 1e-5
factor = 0.
U0 = np.exp(z)
U1 = np.exp(-z) * np.cosh(2*z) 
U2 = np.exp(-z) * np.sinh(2*z)

# The local TEBD algorithm
def tebd(j):
    global factor
    # Create the set of 4 matrices 
    C00 = U0 * la.multi_dot([np.diag(L[j-1]), B[j][0], np.diag(L[j]), B[j+1][0], np.diag(L[j+1])])

    t01 = la.multi_dot([B[j][0], np.diag(L[j]), B[j+1][1]])
    t10 = la.multi_dot([B[j][1], np.diag(L[j]), B[j+1][0]])
    C01 = la.multi_dot([np.diag(L[j-1]), U1 * t01 + U2 * t10, np.diag(L[j+1])])
    C10 = la.multi_dot([np.diag(L[j-1]), U1 * t10 + U2 * t01, np.diag(L[j+1])])

    C11 = U0 * la.multi_dot([np.diag(L[j-1]), B[j][1], np.diag(L[j]), B[j+1][1], np.diag(L[j+1])])

    # Assemble the Q matrix
    upper = np.empty((C00.shape[0], 2 * C00.shape[1]))
    upper[:, 0::2] = C00
    upper[:, 1::2] = C01

    lower = np.empty((C11.shape[0], 2 * C11.shape[1]))
    lower[:, 0::2] = C10
    lower[:, 1::2] = C11

    Q = np.empty((2*upper.shape[0], upper.shape[1]))
    Q[0::2] = upper
    Q[1::2] = lower
    print(Q.shape)

    # Do the SVD, truncate the result
    # print('Calculating SVD...', end=' ', flush=True)
    U, D, V = la.svd(Q, full_matrices=False)
    # print('Done')
    norm = la.norm(D)
    D /= norm
    factor += np.log(norm)
    Mmax = D.size - np.argmax(np.cumsum(np.flip(D**2)) > error)
    U = U[:, :Mmax]
    D = D[:Mmax]
    V = V[:Mmax]

    # Update the MPA
    U = np.dot(np.diag(np.repeat(1/L[j-1], 2)), U)
    B[j][0] = U[0::2]
    B[j][1] = U[1::2]

    L[j] = D

    V = np.dot(V, np.diag(np.repeat(1/L[j+1], 2)))
    B[j+1][0] = V[:, 0::2]
    B[j+1][1] = V[:, 1::2]


# Propagate
for step in range(steps):
    print('Step %d' % step, end='\r')
    # Odd spins
    for j in range(1, n + 1, 2):
        tebd(j)
    # Even spins
    for j in range(2, n + 1, 2):
        tebd(j)
print()
print(np.exp(factor))
