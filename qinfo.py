import numpy as np
from qutip import *

"""
Amplitude damping Kraus operators for a qudit system. Can be used with qutip.mesolve as the c_ops argument.
`dims` is a list of dimensions of qudit subsystems, and t1 is the qubit |1> state coherence time.
Assumes that damping times scale as t1 / n, per blok_quantum_2021 and chessa_quantum_2021.
For more info see https://en.wikipedia.org/wiki/Amplitude_damping_channel
"""
def qudit_amp_damp_ops(t1, dims):
    ks = []
    for n,d in enumerate(dims):
        kij = []
        k0 = basis(d, 0).proj()
        for i in range(d-1):
            j = i+1
            t1_eff = t1 / j
            rate = 1/t1_eff
            kij.append(np.sqrt(rate) * basis(d, i) * basis(d, j).dag())
            k0 += np.sqrt(1-rate) * basis(d, j).proj()
        kij.append(k0)
        for m in range(len(dims)):
            if m < n:
                kij = [tensor(qeye(dims[m]), k) for k in kij]
            if m > n:
                kij = [tensor(k, qeye(dims[m])) for k in kij]
        ks += kij
    return ks