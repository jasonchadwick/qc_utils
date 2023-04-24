# Written by Max Seifert

import numpy as np

def VZ(phases):
    phases = np.concatenate(([0], phases))
    return np.diag(np.exp(1j * np.cumsum(phases)))
    
def graph_VZ(graph, d, phases=None):
    if phases is None:
        phases = graph.optimization_variable(
            d-1, lower_bound=0, upper_bound=2*np.pi, name="phases"
        )
        print("Added optimizable node 'phases'.")

    mat = np.zeros((d,d))
    mat[0,0] = 1
    Zgate = graph.tensor(mat)
    for i in range(d-1):
        mat = np.zeros((d,d))
        mat[i+1,i+1] = 1
        Zgate += graph.exp(1j*graph.sum(phases[:i+1])) * graph.tensor(mat)
    
    return Zgate, phases
    
def target_up_to_VZ(graph, gate, phases=None):
    d = np.shape(gate)[0]
    Zgate, phases = graph_VZ(graph, d, phases)
    return graph.target(operator=graph.adjoint(Zgate)@gate)