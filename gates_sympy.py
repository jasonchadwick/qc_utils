import numpy as np
from sympy import *

def mat(m):
    return Matrix(m)

def rot(m, phi):
    return Matrix(exp(-1j * m * phi))

def u3(th,ph,la):
    return mat([
        [cos(th/2), -exp(1j*la)*sin(th/2)],
        [exp(1j*ph)*sin(th/2), exp(1j*(ph+la))*cos(th/2)]
    ])

def p(th):
    return mat([
        [1,0],
        [0,exp(1j*th)]
    ])

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

CNOTr = mat([
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

a = Symbol('a', real=True)
b = Symbol('b', real=True)
c = Symbol('c', real=True)
d = Symbol('d', real=True)
e = Symbol('d', real=True)
f = Symbol('d', real=True)