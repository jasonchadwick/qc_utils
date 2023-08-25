import qutip as qt
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from functools import reduce
import scipy

def ham_and_ctrls_to_callable(
        H0: qt.Qobj,
        Hctrls: list[qt.Qobj],
        controls: list[Callable[[float], complex]],
    ) -> Callable[[float], qt.Qobj]:
    """Apply time-dependent control functions to a Hamiltonian and return a
    callable function that gives the full Hamiltonian at any point in time.
    
    Args:
        H0: constant drift Hamiltonian.
        Hctrls: list of control Hamiltonians.
        controls: list of control functions, corresponding to Hctrls.
    
    Returns:
        Function mapping time to the instantaneous Hamiltonian that results
        from applying the controls.
    """
    def H(t, args=None):
        H_acc = H0
        for i,Hctrl in enumerate(Hctrls):
            H_acc += controls[i](t) * Hctrl
        return H_acc
    return H

def ff_transmon(
        dims: list[int] | NDArray[np.int_], 
        wr: list[float] | NDArray[np.float_] | None, 
        w0s: list[float] | NDArray[np.float_], 
        deltas: list[float] | NDArray[np.float_], 
        js: NDArray[np.float_], 
    ) -> tuple[qt.Qobj, list[qt.Qobj]]:
    """Return drift Hamiltonian and control Hamiltonian for an N-dimensional
    fixed-frequency transmon system.

    Args:
        dims: number of energy levels to consider for each system (length N).
        wr: rotating-frame frequencies (length N) [2pi*Hz].
        w0s: qubit frequencies (length N) [2pi*Hz].
        deltas: qubit anharmonicities (length N) [2pi*Hz].
        js: qubit-qubit couplings (NxN upper triangular) [2pi*Hz].
    
    Returns:
        Drift and control Hamiltonians.
    """
    rot = True
    if wr is None:
        rot = False
        wr = np.zeros(len(w0s))

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

    H0 = qt.Qobj()
    Hctrls = []
    # create individual qubit Hamiltonian terms
    for i,w0 in enumerate(w0s):
        a = a_ops[i].copy()
        adag = adag_ops[i].copy()
        H0 += (w0-wr[i])*adag*a + deltas[i]/2*adag*adag*a*a
        if rot:
            Hctrls.append(1/2*(adag+a))
            Hctrls.append(1/2*1j*(a-adag))
        else:
            Hctrls = adag+a
        
    # create coupling Hamiltonian terms
    for i in range(nbits):
        for j in range(nbits):
            if j > i:
                H0 += js[i,j]*(adag_ops[i]*a_ops[j] + a_ops[i]*adag_ops[j])

    return H0, Hctrls

def tunable_coupler_hamiltonian(
        qubit_freqs: list[float],
        qubit_anharmonicities: list[float],
        couplers: list[tuple[int, int]],
        coupler_anharmonicities: list[float],
        dims: list[int],
        rot_freqs: list[float],
        coupling_strengths: dict[tuple[int | tuple[int,int], int | tuple[int,int]], float],
        interaction_frame: bool = False,
    ) -> tuple[qt.Qobj, list[qt.Qobj] | list[Callable[[float], qt.Qobj]]]:
    """Build a QuTiP Hamiltonian for a fixed frequency transmon with tunable 
    coupler architecture.

    Note: currently does NOT include single-qubit microwave drives (only 
    coupler tunability).

    Args:
        qubit_freqs: list of frequencies (Hz).
        qubit_anharmonicities: list of anharmonicities (Hz).
        couplers: list of (q0,q1) pairs, where qubits are identified by their 
            index in `qubit_freqs` list.
        coupler_anharmonicities: list of anharmonicities (Hz), of same length 
            as `couplers`.
        dims: list of dimensions to simulate for each oscillator; length is 
            `len(qubits) + len(couplers)`.
        rot_freqs: list of rotating frame frequencies (Hz).
        coupling_strengths: dictionary tracking all couplings between qubits
            and couplers. Can be symmetric, i.e. `coupling_strengths[(x,y)] = coupling_strengths[(y,x)]`, but this is not required.
        interaction_frame: if True, return Hamiltonian in the interaction
            frame.
    
    Returns:
        Drift Hamiltonian and a list of control Hamiltonians (possibly 
        functions of time, if `interaction_frame = True`)
    """
    qubits = list(range(len(qubit_freqs)))
    oscillators = qubits + couplers

    assert len(qubit_freqs) == len(qubit_anharmonicities)
    assert len(couplers) == len(coupler_anharmonicities)
    assert len(dims) == len(oscillators)
    assert len(rot_freqs) == len(dims)

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
    
    # create drift terms
    tot_dims = reduce(lambda x,y:x*y, dims)
    H0 = qt.Qobj(np.zeros((tot_dims, tot_dims)), dims=[dims, dims])
    Hctrl = []
    # create individual qubit Hamiltonian terms
    for i,w0 in enumerate(qubit_freqs):
        a = a_ops[i]
        adag = adag_ops[i]
        H0 += (w0-rot_freqs[i])*adag*a + qubit_anharmonicities[i]/2*adag*adag*a*a
    for i in range(len(couplers)):
        a = a_ops[i+len(qubit_freqs)]
        adag = adag_ops[i+len(qubit_freqs)]
        H0 += -rot_freqs[i+len(qubit_freqs)]*adag*a + coupler_anharmonicities[i]/2*adag*adag*a*a
        Hctrl.append(adag*a) # tunable coupler frequency

    already_generated_pairs = set()
    # create couplings
    for (item0,item1),coupling in coupling_strengths.items():
        assert item0 != item1
        if (item0,item1) not in already_generated_pairs and (item1,item0) not in already_generated_pairs:
            index0 = oscillators.index(item0)
            index1 = oscillators.index(item1)
            a0 = a_ops[index0]
            adag0 = adag_ops[index0]
            a1 = a_ops[index1]
            adag1 = adag_ops[index1]
            H0 += coupling * (adag0 + a0) * (adag1 + a1)
    
    if interaction_frame:
        # TODO: if hctrl has multiple terms, are we allowed to rotate them separately like this?
        return qt.qeye(dims), [lambda t: scipy.linalg.expm(1j*H0*t) @ hctrl @ scipy.linalg.expm(-1j*H0*t) for hctrl in Hctrl]
    
    return H0, Hctrl