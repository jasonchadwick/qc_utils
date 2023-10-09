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
        num_qubits: the number of qubits in the tomography experiment.
        StateType: the type of the qubit state.
        state: the current state of the qubit.

    Usage example (recovering the |0> or |+> state depending on keyword
    arguments):
    ```
    import qiskit
    import qiskit_aer
    import numpy as np
    from qc_utils.tomography.state_tomography import StateTomography
    from qc_utils import density

    class QiskitStateTomography(StateTomography):
        def __init__(self):
            super().__init__(qiskit.QuantumCircuit, 1)
            self.state: qiskit.QuantumCircuit = qiskit.QuantumCircuit(1, 1)
            self.qubit = 0
        
        def initialize(self, state='0', **kwargs):
            '''Initialize qubit into either the 0 or + state depending on the
            keyword argument `state`.
            '''
            self.state = qiskit.QuantumCircuit(1, 1)
            if state == '+':
                self.state.h(self.qubit)

        def measure_X(self, **kwargs):
            self.state.h(self.qubit)
            return self.measure_Z()

        def measure_Y(self, **kwargs):
            self.state.sdg(self.qubit)
            self.state.h(self.qubit)
            return self.measure_Z()

        def measure_Z(self, **kwargs):
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

    def __init__(self, StateType = typing.Any, num_qubits: int = 1):
        """Initialize the StateTomography object.
        
        Args:
            qubits: list of qubits on which we want to perform state
                tomography. If None, assumes that the system consists of only
                a single qubit.
        """
        self.StateType = StateType
        self.state: StateType | None = None
        self.num_qubits = num_qubits

    def initialize(self, **kwargs) -> None:
        """To be implemented by child class.

        This method should initialize self.state to the state of interest.
        """
        raise NotImplementedError

    def measure_X(self, **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the X basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError
    
    def measure_Y(self, **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the Y basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError
    
    def measure_Z(self, **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the Z basis. For a single qubit, the 
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
            measure_X_results = self.measure_X(**kwargs)

            # measure Y
            self.initialize(**kwargs)
            measure_Y_results = self.measure_Y(**kwargs)

            # measure Z
            self.initialize(**kwargs)
            measure_Z_results = self.measure_Z(**kwargs)

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