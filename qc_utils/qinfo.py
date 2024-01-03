import numpy as np
from numpy.typing import NDArray
import qutip as qt

def qudit_amp_damp_ops(t1: float, dims: list[int] | NDArray[np.int_]) -> list[qt.Qobj]:
    """Amplitude damping Kraus operators for a qudit system. Can be used with
    qutip.mesolve as the c_ops argument. Assumes that damping times scale as 
    t1 / n, per blok_quantum_2021 and chessa_quantum_2021. For more info see 
    https://en.wikipedia.org/wiki/Amplitude_damping_channel
    
    Args:
        dims: list of dimensions of qudit subsystems.
        t1: qubit |1> state coherence time.
    
    Returns:
        A list of operators that describe the amplitude damping channels
        in a multi-state (qudit) system.
    """
    ks = []
    for n,d in enumerate(dims):
        kij = []
        k0 = qt.basis(d, 0).proj()
        for i in range(d-1):
            j = i+1
            t1_eff = t1 / j
            rate = 1/t1_eff
            kij.append(np.sqrt(rate) * qt.basis(d, i) * qt.basis(d, j).dag())
            k0 += np.sqrt(1-rate) * qt.basis(d, j).proj()
        kij.append(k0)
        for m in range(len(dims)):
            if m < n:
                kij = [qt.tensor(qt.qeye(dims[m]), k) for k in kij]
            if m > n:
                kij = [qt.tensor(k, qt.qeye(dims[m])) for k in kij]
        ks += kij
    return ks