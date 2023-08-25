import numpy as np
from numpy.typing import NDArray
from qc_utils.idx import *
from qc_utils import matrix

def vec_to_dm(vec: list[complex] | list[list[complex]] | NDArray[np.complex_]) -> NDArray[np.complex_]:
    """Create a density matrix from a statevector.

    Args:
        vec: length `d` statevector. Can be a 1D list e.g. [1,0], a 2D
            column-list e.g. [[1],[0]], or a 2D column numpy array.
    
    Returns:
        A `d`-by-`d` density matrix representing the state.
    """
    vec = np.array(vec, complex)
    if len(vec.shape) == 1:
        # just a list of numbers
        return matrix.adj(np.array([vec])) @ np.array([vec])
    else:
        return vec @ matrix.adj(vec)

def eigenvecs(rho: NDArray[np.complex_]) -> tuple[list[NDArray[np.complex_]], list[NDArray[np.complex_]]]:
    """Decompose a density matrix into its eigenvectors.
    
    Args:
        rho: density matrix.
    
    Returns:
        A list of nonzero eigenvalues, and a list of corresponding eigenvectors.
    """
    evals, evecs = np.linalg.eig(rho)

    evals_nonzero = []
    evecs_nonzero = []

    for eval,evec in zip(list(evals), list(matrix.vec(x) for x in evecs.transpose())):
        if not np.isclose(eval, 0):
            # adjust phase so that first element has 0 angle
            correction_phase = -np.angle(evec[0])
            evals_nonzero.append(eval * np.exp(1j*correction_phase))
            evecs_nonzero.append(evec * np.exp(1j*correction_phase))

    return evals_nonzero, evecs_nonzero

def nearest_pure_state(rho: NDArray[np.complex_]) -> NDArray[np.complex_]:
    """Find the pure state that contributes the most to the density matrix rho.
    
    Args:
        rho: density matrix to analyze.
    
    Returns:
        A pure statevector that is closest to rho.
    """
    return eigenvecs(rho)[1][0]

def schmidt_decomp(v: NDArray[np.complex_], n0: int, n1: int) -> tuple[list[float], list[NDArray[np.complex_]], list[NDArray[np.complex_]]]:
    """Perform Schmidt decomposition on input statevector into a sum of pure
    states of two subsytems.
    
    Args:
        v: statevector to decompose.
        n0: dimension of subsystem 0.
        n1: dimension of subsystem 1.
    
    Returns:
        List of weights and lists of associated statevectors in the two
        subsystems.
    """
    mat = np.zeros((n0,n1), complex)
    for idx in range(len(v)):
        bits = bits_from_idx(idx, [n0,n1])
        mat[bits[0], bits[1]] = v[idx]
    u1,d,u2 = np.linalg.svd(mat)

    c = []
    a = []
    b = []

    for i,val in enumerate(d):
        if not np.isclose(0, val):
            c.append(val)
            a.append(matrix.vec(u1[:,i]))
            b.append(matrix.vec(u2[i,:]))

    return c,a,b

def schmidt_num(v: NDArray[np.complex_], n0: int, n1: int) -> int:
    """Count the number of pure states in the Schmidt decomposition of the 
    input statevector.
    
    Args:
        v: statevector to decompose.
        n0: dimension of subsystem 0.
        n1: dimension of subsystem 1.
    
    Returns:
        List of weights and lists of associated statevectors in the two
        subsystems.
    """
    return len(schmidt_decomp(v,n0,n1)[0])

def apply_chi_channel(rho: NDArray[np.complex_], chi_matrix: NDArray[np.complex_], chi_basis_elems: list[NDArray[np.complex_]]) -> NDArray[np.complex_]:
    """Apply a quantum channel that is described by a chi matrix and a list of
    basis elements E_i, as can be obtained by quantum process tomography.
    
    Args:
        rho: state (density matrix) to apply process to.
        chi_matrix: matrix characterizing the process.
        chi_basis_elems: basis elements of process, which together with
            chi_matrix define the process.
    
    Returns:
        rho_new: state after applying the process.
    """
    assert chi_matrix.shape == (len(chi_basis_elems), len(chi_basis_elems))
    assert rho.shape == chi_basis_elems[0].shape

    rho_new = np.zeros(rho.shape, complex)
    for m,E_m in enumerate(chi_basis_elems):
        for n,E_n in enumerate(chi_basis_elems):
            rho_new += E_m @ rho @ E_n.T.conj() * chi_matrix[m,n]
    return rho_new