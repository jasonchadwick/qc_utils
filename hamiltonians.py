import qutip as qt
import numpy as np

def ff_transmon(dims, wr, w0s, deltas, js, controls=None):
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

    rot = True
    if wr is None:
        rot = False
        wr = np.zeros(len(w0s))
    if type(wr) != list and type(wr) != np.ndarray:
        wr = [wr]
    if type(dims) != list and type(dims) != np.ndarray:
        dims = [dims]
    if type(w0s) != list and type(w0s) != np.ndarray:
        w0s = [w0s]
    if type(deltas) != list and type(deltas) != np.ndarray:
        deltas = [deltas]

    dims = np.array(dims)
    nbits = len(dims)

    # create qubit operators
    a_ops = []
    adag_ops = []
    for i,d in enumerate(dims):
        kron_before = [qt.identity(d) for d in  dims[:i]]
        kron_after = []
        if i < len(dims)-1:
            kron_after = [qt.identity(d) for d in  dims[i+1:]]
            
        a_ops.append(qt.tensor(kron_before + [qt.destroy(d)] + kron_after))
        adag_ops.append(qt.tensor(kron_before + [qt.create(d)] + kron_after))

    H0 = 0
    Hctrls = []
    # create individual qubit Hamiltonian terms
    for i,w0 in enumerate(w0s):
        a = a_ops[i].copy()
        adag = adag_ops[i].copy()
        H0 += (w0-wr[i])*adag*a + deltas[i]/2*adag*adag*a*a
        ctrls = []
        if rot:
            ctrls.append(1/2*(adag+a))
            ctrls.append(1/2*1j*(a-adag))
        else:
            ctrls = adag+a
        Hctrls.append(ctrls)
        
    # create coupling Hamiltonian terms
    for i in range(nbits):
        for j in range(nbits):
            if j > i:
                H0 += js[i,j]*(adag_ops[i]*a_ops[j] + a_ops[i]*adag_ops[j])
    
    if controls is not None:
        if rot:
            for i in range(nbits):
                for j in range(2):
                    if controls[i][j] is None:
                        controls[i][j] = lambda t : 0

            def H(t, args=None):
                H_acc = 0
                H_acc += H0
                for i in range(nbits):
                    H_acc += controls[i][0](t) * Hctrls[i][0]
                    H_acc += controls[i][1](t) * Hctrls[i][1]
                return H_acc
            return H
        else:
            for i in range(nbits):
                if controls[i] is None:
                    controls[i] = lambda t : 0

            def H(t, args=None):
                H_acc = 0
                H_acc += H0
                for i in range(nbits):
                    H_acc += controls[i](t) * Hctrls[i]
                return H_acc
            return H

    return H0, Hctrls