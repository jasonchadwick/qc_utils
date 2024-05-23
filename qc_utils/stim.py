import stim
import itertools

def depolarize(
            stim_circ: stim.Circuit,
            qubits: list[int],
            p: float,
        ) -> stim.Circuit:
        """Apply a multi-qubit correlated depolarizing error to a Stim circuit.
        
        Applies each possible Pauli error (including identity) to the qubits
        with equal probability.

        Args:
            stim_circ: Stim circuit to append to.
            qubits: List of qubits to apply the operation to.
            p: Total probability of error.

        Returns:
            Modified Stim circuit with the error appended.
        """
        if len(qubits) == 1:
            stim_circ.append('DEPOLARIZE1', qubits, p*3/4)
            return stim_circ
        elif len(qubits) == 2:
            stim_circ.append('DEPOLARIZE2', qubits, p*15/16)
            return stim_circ

        paulis = ['I', 'X', 'Y', 'Z']
        targets = {'X':stim.target_x, 'Y':stim.target_y, 'Z':stim.target_z}

        independent_prob = p / 4**len(qubits)

        skip_identity = True
        first_error = True
        previous_probs = []
        for paulis in itertools.product(paulis, repeat=len(qubits)):
            if skip_identity:
                # skip the first iteration, which is the identity
                skip_identity = False
                continue

            target_list = []
            for i,pauli in enumerate(paulis):
                if pauli != 'I':
                    target_list.append(targets[pauli](qubits[i]))
            if first_error:
                prob = independent_prob
                stim_circ.append('CORRELATED_ERROR', target_list, prob)
                first_error = False
                previous_probs.append(prob)
            else:
                prob = float(independent_prob / np.prod(1 - np.array(previous_probs)))
                stim_circ.append('ELSE_CORRELATED_ERROR', target_list, prob)
                previous_probs.append(prob)
        return stim_circ