import qutip as qt
import numpy as np

def ff_transmon(graph, wr, dims, w0s, deltas, js, controls=None):
    """
    Return drift Hamiltonian and control Hamiltonian for an N-dimensional fixed-frequency transmon system.
    If `controls` is given, return time-dependent Hamiltonian. Otherwise, return a tuple (H0,Hctrl),
    where Hctrl is an Nx2 list containing the X and Y control Hamiltonians for each subsystem.

    Parameters:
    `wr` rotating-frame frequencies (list{float}, N) [2pi*Hz]
    `dims` number of energy levels to consider for each system (list{int}, N)
    `w0s` qubit frequencies (list{float}, N) [2pi*Hz]
    `deltas` qubit anharmonicities (list{float}, N) [2pi*Hz]
    `js` qubit-qubit couplings (np.ndarray{float}, NxN upper triangular) [2pi*Hz]
    `controls` real-valued X and Y controls (Real(Omega) and Imag(Omega), functions of time), if given. Assumed to be in rotating frame already.
    """

    dims = np.array(dims)
    nbits = len(dims)

    # create qubit operators
    a_ops = []
    adag_ops = []
    for i,d in enumerate(dims):
        kron_before = [np.identity(d) for d in  dims[:i]]
        kron_after = []
        if i < len(dims)-1:
            kron_after = [np.identity(d) for d in  dims[i+1:]]
            
        a_ops.append(graph.kronecker_product_list(kron_before + [graph.annihilation_operator(d)] + kron_after))
        adag_ops.append(graph.kronecker_product_list(kron_before + [graph.creation_operator(d)] + kron_after))

    H0 = 0
    Hctrls = []
    # create individual qubit Hamiltonian terms
    for i,w0 in enumerate(w0s):
        a = a_ops[i]
        adag = adag_ops[i]
        H0 += (w0-wr[i])*adag*a + deltas[i]/2*adag*adag*a*a
        ctrls = []
        ctrls.append(1/2*(adag+a))
        ctrls.append(1/2*1j*(a-adag))
        Hctrls.append(ctrls)
        
    # create coupling Hamiltonian terms
    for i in range(nbits):
        for j in range(nbits):
            if j > i:
                H0 += js[i,j]*(adag_ops[i]*a_ops[j] + a_ops[i]*adag_ops[j])
    
    if controls is not None:
        H = 0
        H += H0
        for i in range(nbits):
            H += controls[i][0] * Hctrls[i][0]
            H += controls[i][1] * Hctrls[i][1]
        return H

    return H0, Hctrls