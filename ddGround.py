import numpy as np
from useful import gstate

grounds = []
for n in range(2, 21, 2):
    grounds.append(gstate(n, False))
np.save('data/ddGround.npy', grounds)
