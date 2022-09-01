from scipy import linalg
import numpy as np
from functools import reduce
from itertools import product
from .matrix import *

def is_unitary(m):
    return m.shape[0] == m.shape[1] and np.all(np.isclose(adj(m) @ m, np.identity(m.shape[0])))

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

def fid(u1, u2):
    return np.abs(np.trace(np.transpose(np.conjugate(u1)) @ u2))**2 / u1.shape[0]**2

def pauli_decomp(u):
    nbits = int(np.log2(u.shape[0]))
    coeffs = []
    for basis in product((X,Y,Z,i2), repeat=nbits):
        basis_mat = reduce(np.kron, basis)
        coeff = 1/4 * np.trace(basis_mat @ u)
        coeffs.append(coeff)
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