import numpy as np
from matplotlib import pyplot as plt

nes = [10, 20, 50, 80, 100, 120, 200]

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

data = []
lines = []
names = []
for n in nes:
    data.append(np.load('data/energies-%d.npy' % n))
    lines.append(ax.plot(data[-1][0], data[-1][1]/n)[0])
    names.append('$n = %d$' % n)

print(data[-2][1][-1]/120)

ax.grid()
ax.set_xscale('log')
ax.set_ylabel('$E/n$')
ax.set_xlabel('$\\beta$')
ax.legend(lines, names)
plt.show()

