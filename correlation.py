import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

n = 100
mpa = np.load('data/mpa-%d.npy' % n)
B = mpa[0]
L = mpa[1]

T = []
V = []
for j in range(1, n + 1):
    A0 = np.dot(np.diag(L[j-1]), B[j][0])
    A1 = np.dot(np.diag(L[j-1]), B[j][1])
    P0 = np.kron(A0, A0)
    P1 = np.kron(A1, A1)
    T.append(P0 + P1)
    V.append(P0 - P1)

norm = la.multi_dot(T).reshape(())

C = np.empty((n, n))
for j in range(0, n):
    C[j, j] = 1.0
    for k in range(j + 1, n):
        print('Row %d Col %d' % (j, k), end='\r')
        C[j, k] = la.multi_dot(T[:j] + [V[j]] + T[j+1:k] + [V[k]] + T[k + 1:]).reshape(()) / norm
        C[k, j] = C[j, k]
print()

np.save('data/corr-%d.npy' % n, C)

plt.imshow(C)
plt.show()
