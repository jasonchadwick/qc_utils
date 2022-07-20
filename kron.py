import numpy as np
    
def rev_kron(mat, submat, known_pos, qubit=0):
    newmat = np.zeros((mat.shape[0] // submat.shape[0], mat.shape[1] // submat.shape[1]))
    for r in range(newmat.shape[0]):
        for c in range(newmat.shape[1]):
            if known_pos == 0:
                newmat[r,c] = mat[r,c] / submat[0,0]
            else:
                newmat[r,c] = mat[r*submat.shape[0], c*submat.shape[1]] / submat[0,0]