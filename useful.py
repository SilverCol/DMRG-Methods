import numpy as np
import scipy.sparse as sp

def kronH(N, periodic):
    """ Constructs heisenberg spin chain Hamiltonian using Kronecker products. """

    # Initialize constants
    DIM = 2**N
    S2 = sp.csc_matrix([
        [1,  0,  0, 0],
        [0, -1,  2, 0],
        [0,  2, -1, 0],
        [0,  0,  0, 1]
        ])

    # Hamiltonian matrix
    H = sp.csc_matrix((DIM, DIM))
    for j in range(1, N):
        print('Term %d/%d' % (j, N), end='\r')
        H += sp.kron( sp.kron(sp.identity(2**(j-1)), S2), sp.identity(2**(N - j - 1)) )
    print()

    if periodic:
        Sx = sp.csc_matrix([
            [0, 1],
            [1, 0]
            ])
        H += sp.kron( sp.kron(Sx, sp.identity(2**(N-2))), Sx )

        Sy = sp.csc_matrix([
            [0, -1j],
            [1j,  0]
            ])
        H += np.real( sp.kron( sp.kron(Sy,sp.identity(2**(N-2))), Sy ) )

        Sz = sp.csc_matrix([
            [1,  0],
            [0, -1]
            ])
        H += sp.kron( sp.kron(Sz, sp.identity(2**(N-2))), Sz )
    return H

from scipy.sparse.linalg import eigsh

def gstate(N, periodic):
    """ Creates a spin chain ground state, saves it to a file. """

    # Create Hamiltonian matrix
    H = kronH(N, periodic)
   
    # Diagonalize
    print('Diagonalizing...', end=' ', flush=True)
    w, v = eigsh(H, k=1, which='SA')
    print('Done')

    return w[0]
