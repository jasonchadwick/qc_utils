import numpy as np
from numpy.typing import NDArray
import typing
import qc_utils.gates as gates

class StateTomography():
    """Class used to define a general state tomography experiment, without
    assuming anything about the type or behavior of the state representation.
    
    Requires user to define the functions that relate to initializing or
    operating on the state.

    Currently limited to a single logical qubit.

    Attributes:
        qubits: the indices of the qubits on which to perform tomography.
        num_qubits: the number of qubits in the tomography experiment.
        StateType: the type of the qubit state.
        state: the current state of the qubit.

    Usage example (recovering the |0> or |+> state depending on keyword
    arguments):
    ```
    from qc_utils.tomography.state_tomography import StateTomography
    from qc_utils import density

    class QiskitStateTomography(StateTomography):
        def __init__(self):
            super().__init__(0, qiskit.QuantumCircuit)
            self.state: qiskit.QuantumCircuit = qiskit.QuantumCircuit(1, 1)
        
        def initialize(self, state='0', **kwargs):
            '''Initialize qubit into either the 0 or + state depending on the
            keyword argument `state`.
            '''
            self.state = qiskit.QuantumCircuit(1, 1)
            if state == '+':
                self.state.h(0)

        def H_op(self, tgts, **kwargs):
            self.state.h(tgts)
        
        def Sdg_op(self, tgts, **kwargs):
            self.state.sdg(tgts)
        
        def measure(self, **kwargs):
            # simulate in Qiskit
            backend = qiskit_aer.StatevectorSimulator()
            results = backend.run(self.state).result().results[0].data.statevector
            return [np.abs(results[0])**2, np.abs(results[1])**2]
    
    qst = QiskitStateTomography()
    dm = qst.run_state_tomography(state='+')

    print(density.nearest_pure_state(dm))
    # should print "array([[0.70710678], [0.70710678]])"
    ```
    """

    def __init__(self, qubits: int | list[int] = 0, StateType = typing.Any):
        """Initialize the StateTomography object.
        
        Args:
            qubits: list of qubits on which we want to perform state
                tomography. If None, assumes that the system consists of only
                a single qubit.
        """
        self.tgt_qubits: int | list[int] = qubits
        if qubits is None or type(qubits) is int:
            self.num_qubits = 1
        else:
            assert type(qubits) is list[int]
            self.num_qubits = len(qubits)
        self.StateType = StateType
        self.state: StateType | None = None

    def initialize(self, **kwargs) -> None:
        """To be implemented by child class.

        This method should initialize self.state to the state of interest.
        """
        raise NotImplementedError

    def H_op(self, qubits: int | list[int], **kwargs) -> None:
        """To be implemented by child class.

        This method should apply an inplace logical X operation to self.state
        on the qubits specified by `qubits`.

        Args:
            qubits: the qubit indices on which to apply the operation.
        """
        raise NotImplementedError

    def Sdg_op(self, qubits: int | list[int], **kwargs) -> None:
        """To be implemented by child class.

        This method should apply an inplace logical X operation to self.state
        on the qubits specified by `qubits`.

        Args:
            qubits: the qubit indices on which to apply the operation.
        """
        raise NotImplementedError

    def measure(self, qubits: int | list[int], **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the computational (Z) basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError

    def run_state_tomography(self, **kwargs) -> NDArray[np.complex_]:
        """Requires that methods implementing initialization, logical
        operations, and measurement have been properly defined.
        
        Args:
            kwargs: arguments to pass into user-defined functions.

        Returns:
            A density matrix representing the tomography measurement of the
            state.
        """
        if self.num_qubits == 1:
            # measure X
            self.initialize(**kwargs)
            self.H_op(self.tgt_qubits, **kwargs)
            measure_X_results = self.measure(self.tgt_qubits, **kwargs)

            # measure Y
            self.initialize(**kwargs)
            self.Sdg_op(self.tgt_qubits, **kwargs)
            self.H_op(self.tgt_qubits, **kwargs)
            measure_Y_results = self.measure(self.tgt_qubits, **kwargs)

            # measure Z
            self.initialize(**kwargs)
            measure_Z_results = self.measure(self.tgt_qubits, **kwargs)

            xbar = measure_X_results[0] - measure_X_results[1]
            ybar = measure_Y_results[0] - measure_Y_results[1]
            zbar = measure_Z_results[0] - measure_Z_results[1]

            norm = xbar**2 + ybar**2 + zbar**2

            if norm > 1:
                xbar /= norm
                ybar /= norm
                zbar /= norm
            
            output_dm = 1/2 * (gates.i2 + xbar*gates.X + ybar*gates.Y + zbar*gates.Z)

            return output_dm
        else:
            # currently only supports one qubit
            raise NotImplementedError