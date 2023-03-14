import qutip as qt
import numpy as np
from functools import reduce

def transmon(wr, dims, w0s, deltas, js):
    """
    Return drift Hamiltonian and control Hamiltonian for an N-dimensional transmon system.

    Parameters:
    `wr` rotating-frame frequencies (list{float}, N) [2pi*Hz]
    `dims` number of energy levels to consider for each system (list{int}, N)
    `w0s` qubit frequencies (list{float}, N) [2pi*Hz]
    `deltas` qubit anharmonicities (list{float}, N) [2pi*Hz]
    `js` qubit-qubit couplings (np.ndarray{float}, NxN upper triangular) [2pi*Hz]
    """

    dims = list(dims)
    dims = np.array(dims)

    # create qubit operators
    a_ops = []
    adag_ops = []
    for i,d in enumerate((dims)):
        kron_before = [qt.identity(d) for d in  dims[:i]]
        kron_after = []
        if i < len(dims)-1:
            kron_after = [qt.identity(d) for d in  dims[i+1:]]
            
        a_ops.append(qt.tensor(kron_before + [qt.destroy(d)] + kron_after))
        adag_ops.append(qt.tensor(kron_before + [qt.create(d)] + kron_after))

    H0 = 0
    Hctrls = [[0,0]] * len(dims)
    # create individual qubit Hamiltonian terms
    for i,w0 in enumerate(w0s):
        a = a_ops[i]
        adag = adag_ops[i]
        H0 += (w0-wr[i])/2*adag*a + deltas[i]/2*adag*adag*a*a
        Hctrls[i][0] = 1/2*(adag+a)
        Hctrls[i][1] = 1/2*1j*(adag-a)

    # create coupling Hamiltonian terms
    for i in range(len(dims)):
        for j in range(len(dims)):
            if j < i:
                H0 += js[i,j]*(adag_ops[i]*a_ops[j] + a_ops[i]*adag_ops[j])
    
    return H0, Hctrls