import numpy as np

# input: a list
def vec(vals):
    return np.array([[x] for x in vals], dtype=complex)

def adj(m):
    return np.conjugate(np.transpose(m))