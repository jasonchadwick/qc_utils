from scipy import linalg
import numpy as np
from functools import reduce
from itertools import product
from .matrix import *
from .idx import idx_from_bits, bits_from_idx
from qiskit import quantum_info

def extend(m, dims):
    total_dims = m.shape[0] + dims
    g = np.eye(total_dims, dtype=complex)
    g[:m.shape[0], :m.shape[1]] = m
    return g

def extend(m, old_dims, added_dims):
    assert(m.shape[0] == reduce(lambda x,y : x*y, old_dims))
    old_dims = np.array(old_dims)
    added_dims = np.array(added_dims)
    dims = old_dims + added_dims
    new_size = reduce(lambda x,y : x*y, dims)
    mat = np.zeros((new_size, new_size), complex)
    for row in range(new_size):
        for col in range(new_size):
            row_bits = bits_from_idx(row, dims)
            col_bits = bits_from_idx(col, dims)
            if np.all(row_bits < old_dims) and np.all(col_bits < old_dims):
                old_row_idx = idx_from_bits(row_bits, old_dims)
                old_col_idx = idx_from_bits(col_bits, old_dims)
                mat[row, col] = m[old_row_idx, old_col_idx]
            else:
                if row == col:
                    mat[row, col] = 1
                else:
                    mat[row, col] = 0
    return mat

def switch_bits(unitary):
    u_new = np.zeros(unitary.shape, complex)
    for row in range(unitary.shape[0]):
        for col in range(unitary.shape[1]):
            new_row = (row % 2)*2 + (row // 2)
            new_col = (col % 2)*2 + (col // 2)
            u_new[new_row, new_col] = unitary[row, col]
    return u_new

def is_unitary(m):
    return m.shape[0] == m.shape[1] and np.all(np.isclose(adj(m) @ m, np.identity(m.shape[0])))

def weyl_coords(m):
    decomp = quantum_info.synthesis.two_qubit_decompose.TwoQubitWeylDecomposition(m)
    a = decomp.a * 2 / np.pi
    b = decomp.b * 2 / np.pi
    c = decomp.c * 2 / np.pi
    return a,b,c

def weyl_decompose(m):
    decomp = quantum_info.synthesis.two_qubit_decompose.TwoQubitWeylDecomposition(m)
    a = decomp.a * 2 / np.pi
    b = decomp.b * 2 / np.pi
    c = decomp.c * 2 / np.pi
    global_phase = decomp.global_phase
    K1l = decomp.K1l
    K2l = decomp.K2l
    K1r = decomp.K1r
    K2r = decomp.K2r
    return a,b,c,global_phase,K1l,K2l,K1r,K2r

def mat(m):
    return np.array(m, complex)

def rot(m, phi):
    return linalg.expm(-1j * m * phi)

def u3(th,ph,la):
    return mat([
        [np.cos(th/2), -np.exp(1j*la)*np.sin(th/2)],
        [np.exp(1j*ph)*np.sin(th/2), np.exp(1j*(ph+la))*np.cos(th/2)]
    ])

def random_u(nbits):
    a = np.random.random((2**nbits,2**nbits)) + 1j*np.random.random((2**nbits,2**nbits))
    u,_ = np.linalg.qr(a)
    return u

X = mat([[0, 1], [1, 0]])
Y = mat([[0, -1j], [1j, 0]])
Z = mat([[1, 0], [0, -1]])
r2 = 1/np.sqrt(2)
H = mat([[r2, r2], [r2, -r2]])
i2 = mat([[1, 0], [0, 1]])

def get_weyl_matrix(a,b,c):
    return linalg.expm(1j*np.pi/2*(a*np.kron(X,X) + b*np.kron(Y,Y) + c*np.kron(Z,Z)))

def rx(phi):
    return rot(X, phi/2)
def ry(phi):
    return rot(Y, phi/2)
def rz(phi):
    return rot(Z, phi/2)

CNOT = mat([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])

CNOT_alt = mat([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
])

CZ = mat([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1]
])

SWAP = mat([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
])

# CNOT = rot(kron(Z,i2),-np.pi/4) @ CR @ rot(kron(i2,X),-np.pi/4) up to a global phase
CR = 1/np.sqrt(2) * mat([
    [1, -1j, 0, 0],
    [-1j, 1, 0, 0],
    [0, 0, 1, 1j],
    [0, 0, 1j, 1]
])

# from https://quantumcomputing.stackexchange.com/questions/16256/what-is-the-procedure-of-finding-z-y-decomposition-of-unitary-matrices
def zy(u):
    a = u[0,0]
    b = u[0,1]
    c = u[1,0]
    d = u[1,1]

    if abs(a) < 1e-5:
        gamma = np.pi
    else:
        gamma = 2*np.arctan(np.abs(b) / np.abs(a))

    # if possible, make beta = delta or beta = -delta so they can cancel out in parts of ABC decomp.
    if abs(gamma) < 1e-5:
        beta = (np.angle(d) - np.angle(a)) / 2
        delta = beta
    elif np.isclose(gamma, np.pi):
        beta = (np.angle(-b) - np.angle(c)) / 2
        delta = -beta
    else:
        beta = np.angle(c) - np.angle(a)
        delta = np.angle(-b) - np.angle(a)

    if abs(a) < 1e-5:
        alpha = np.angle(c) - beta/2 + delta/2
    else:
        alpha = np.angle(a) + beta/2 + delta/2

    return alpha,beta,gamma,delta

def zy_mat(alpha, beta, gamma, delta):
    return np.exp(1j*alpha) * rz(beta) @ ry(gamma) @ rz(delta)

# gives a,A,B,C such that mat = exp(i a) * A @ X @ B @ X @ C, and A @ B @ C = I
# for deconstructing a controlled single-qubit U gate
def ABC(mat):
    alpha,beta,gamma,delta = zy(mat)
    A = rz(beta) @ ry(gamma/2)
    B = ry(-gamma/2) @ rz(-(delta+beta)/2)
    C = rz((delta-beta)/2)
    return alpha, A, B, C

# TODO: define custom gate object (that acts like np array whenever needed), and fidelity of mismatched gates uses closest_unitary to compute
def fid(u1, u2):
    return np.abs(np.trace(u1.T.conj() @ u2))**2 / u1.shape[0]**2

def pauli_decomp(u):
    nbits = int(np.log2(u.shape[0]))
    coeffs = []
    for basis in product((i2,X,Y,Z), repeat=nbits):
        basis_mat = reduce(np.kron, basis)
        coeff = 1/nbits**2 * np.trace(basis_mat @ u)
        coeffs.append(coeff)
        if coeff > 0:
            print(coeff, basis_mat)
    return coeffs
    
def pauli_reconstruct(coeffs, nbits):
    acc = np.zeros((2**nbits, 2**nbits), complex)
    for i,basis in enumerate(product((X,Y,Z,i2), repeat=nbits)):
        basis_mat = reduce(np.kron, basis)
        acc += coeffs[i] * basis_mat
    return acc

def pauli_basis(nbits):
    mats = product([i2, X, Y, Z], repeat=nbits)
    return [reduce(np.kron, ms) for ms in mats]

def closest_unitary(A, Nkeep=None, Nt=None):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to matrix A.

        Return U as a numpy matrix.

        Nkeep: if given, list of dimensions of each subsystem to keep
        Nt: if given, list of total dimensions of each subsystem. Must have A.shape[0] = A.shape[1] = reduce(*, Nt)
    """
    if Nkeep is None:
        Nkeep = [A.shape[0]]
        Nt = [A.shape[0]]
    d_keep = reduce(lambda x,y:x*y, Nkeep)
    d_tot = reduce(lambda x,y:x*y, Nt)
    assert(d_tot == A.shape[0] and d_tot == A.shape[1])
    new_U = np.zeros((d_keep, d_keep), complex)
    idxs_to_keep = []
    for i in range(d_tot):
        bits = bits_from_idx(i, Nt)
        if np.all(bits < np.array(Nkeep)):
            idxs_to_keep.append(i)
    for i in idxs_to_keep:
        for j in idxs_to_keep:
            new_i = idx_from_bits(bits_from_idx(i, Nt), Nkeep)
            new_j = idx_from_bits(bits_from_idx(j, Nt), Nkeep)
            new_U[new_i, new_j] = A[i, j]
    V, __, Wh = linalg.svd(new_U)
    U = np.matrix(V.dot(Wh))
    return U