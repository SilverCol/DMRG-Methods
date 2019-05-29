import numpy as np
from matplotlib import pyplot as plt

nes = range(4, 21, 4)

ddGround = np.load('data/ddGround.npy')
data = []
for n in nes:
    data.append(np.load('data/energies-%d.npy' % n))

    
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.set_xscale('log')
ax.set_ylabel('$E$')
ax.set_xlabel('$\\beta$')

lines = []
names = []
for i, n in enumerate(nes):
    lines.append(ax.plot(data[i][0], data[i][1])[0])
    ax.plot([data[i][0][0], data[i][0][-1]], [ddGround[int(n/2) - 1], ddGround[int(n/2) - 1]], 'k--', lw=1)
    names.append('$n = %d$' % n)

ax.legend(lines, names)

plt.show()
