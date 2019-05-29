import numpy as np
from matplotlib import pyplot as plt

n = 100
C = np.load('data/corr-%d.npy' % n)

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.set_ylabel('$j$')
ax.set_xlabel('$k$')
ax.set_title('$n = %d$' % n)
image = ax.imshow(C)
fig.colorbar(image)
plt.show()
