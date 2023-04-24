import numpy as np
from .idx import *
from .matrix import *

# decomposes into eigenvectors
def eigenvecs(rho):
    evals, evecs = np.linalg.eig(rho)

    evals_nonzero = []
    evecs_nonzero = []

    for eval,evec in zip(list(evals), list(vec(x) for x in evecs.transpose())):
        if not np.isclose(eval, 0):
            evals_nonzero.append(eval)
            evecs_nonzero.append(evec)

    return evals_nonzero, evecs_nonzero

def schmidt_decomp(v, n0, n1):
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
            a.append(vec(u1[:,i]))
            b.append(vec(u2[i,:]))

    return c,a,b

def schmidt_num(v, n0, n1):
    return len(schmidt_decomp(v,n0,n1))[0]