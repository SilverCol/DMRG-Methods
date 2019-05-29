from statistics import stdev, mean
import numpy as np
from matplotlib import pyplot as plt

option = 2
n = 100
C = np.load('data/corr-%d.npy' % n)


plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()

points = [[] for i in range(n)]
for j in range(n):
    points[0].append(1.0)
    for k in range(j + 1, n):
        points[k - j].append(C[j, k])

devs = [stdev(points[r])/abs(mean(points[r])) for r in range(n - 1)]

if option == 0:
    ax.set_ylabel('$C(0, k)$')
    ax.set_xlabel('$k$')
    ax.plot(C[0])
elif option == 1:
    ax.set_ylabel('$\\sigma (C(r)) / \\langle C(r) \\rangle$')
    ax.set_xlabel('$r$')
    ax.plot(devs)
else:
    r = 1
    ax.set_ylabel('$C(j, j + %d)$' % r)
    ax.set_xlabel('$j$')
    ax.plot(range(1, n - r + 1), points[r])

plt.show()
