# Originally written by Max Seifert. I have added/extended parts of it.

import numpy as np
from functools import reduce

def VZ(phases):
    """
    Return a Z operation with the specified relative phases.
    """
    phases = np.concatenate(([0], phases))
    return np.diag(np.exp(1j * np.cumsum(phases)))
    
def graph_VZ(graph, dim, phases=None):
    """
    Create a virtual Z operation in a Q-CTRL graph.

    If `phases` are not given, they are specified as optimizable variables.

    Returns the VZ operator and the set of phases.
    """
    if type(dim) is int:
        if phases is None:
            phases = graph.optimization_variable(
                dim-1, lower_bound=0, upper_bound=2*np.pi, name="phases"
            )
            print("Added optimizable node 'phases'.")

        mat = np.zeros((dim,dim))
        mat[0,0] = 1
        Zgate = graph.tensor(mat)
        for i in range(dim-1):
            mat = np.zeros((dim,dim))
            mat[i+1,i+1] = 1
            Zgate += graph.exp(1j*graph.sum(phases[:i+1])) * graph.tensor(mat)
        
        return Zgate, phases
    else:
        # Make single-qubit VZ gates through self-calls
        
        if phases is None:
            phases = [None] * len(dim)
        full_gate = np.eye(reduce(lambda x,y : x*y, dim))
        full_phases = []
        for i,d in enumerate(dim):
            Zgate, phases_i = graph_VZ(graph, d, phases[i])
            phases_i.name = f'phases_{i}' # avoid duplicate names in graph
            Zgate = graph.kronecker_product_list([np.eye(di) for di in dim[:i]] + [Zgate] + ([np.eye(di) for di in dim[i+1:]] if i < len(dim)-1 else []))
            full_gate = full_gate @ Zgate # order of multiplications does not matter because all gates are diagonal
            full_phases.append(phases_i)
        return full_gate, full_phases
    
def target_up_to_VZ(graph, gate, phases=None, dims=None):
    """
    If phases and dims are provided, phases must be a list of size N 
    (where N = len(dims)) and each entry is a list of size dims[i].
    """
    assert(gate.shape[0] == reduce(lambda x,y : x*y, dims))
    if dims is None:
        dims = np.shape(gate)[0]
    Zgate, _ = graph_VZ(graph, dims, phases)
    return graph.target(operator=graph.adjoint(Zgate)@gate)