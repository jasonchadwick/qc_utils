import numpy as np
from numpy.typing import NDArray
from qc_utils.tomography.state_tomography import StateTomography
import typing
import qc_utils.gates as gates

class StateTomographyHelper(StateTomography):
    def __init__(self, StateType, num_qubits, initialize_fn, measure_X_fn, measure_Y_fn, measure_Z_fn):
        super().__init__(StateType, num_qubits)
        self.initialize_fn = initialize_fn
        self.measure_X_fn = measure_X_fn
        self.measure_Y_fn = measure_Y_fn
        self.measure_Z_fn = measure_Z_fn
    
    def initialize(self, **kwargs):
        return self.initialize_fn(**kwargs)
    
    def measure_X(self, **kwargs):
        return self.measure_X_fn(**kwargs)
    
    def measure_Y(self, **kwargs):
        return self.measure_Y_fn(**kwargs)
    
    def measure_Z(self, **kwargs):
        return self.measure_Z_fn(**kwargs)

class ProcessTomography():
    """Class used to define a general process tomography experiment, without
    assuming anything about the type or behavior of the state representation.
    
    Requires user to define the functions that relate to operating on the 
    system.

    Currently limited to a single logical qubit.
            
    Attributes:
        num_qubits: the number of qubits in the tomography experiment.
        StateType: the type of the qubit state.
        state: the current state of the qubit.
    
    Usage example (recovering an R_X gate):
    ```
    import qiskit
    import qiskit_aer
    import numpy as np
    from qc_utils.tomography.process_tomography import ProcessTomography
    from qc_utils import density

    class QiskitProcessTomography(ProcessTomography):
        def __init__(self):
            super().__init__(qiskit.QuantumCircuit)
            self.state: qiskit.QuantumCircuit = qiskit.QuantumCircuit(1)
            self.qubit = 0
        
        def apply_process(self, **kwargs):
            self.state.rx(1, self.qubit)

        def initialize(self, state, **kwargs):
            self.state.reset(self.qubit)
            if state == '1':
                self.state.x(self.qubit)
            if state == '+' or state == 'i':
                self.state.h(self.qubit)
            if state == 'i':
                self.state.s(self.qubit)

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

    qpt = QiskitProcessTomography()
    chi = qpt.run_process_tomography()
    chi_basis_elems = qpt.chi_matrix_basis_elements()

    rho_init = density.vec_to_dm([1,0])
    rho_out = density.apply_chi_channel(rho_init, chi, chi_basis_elems)
    print(density.nearest_pure_state(rho_out))
    # should print "[0.87758256, -0.47942554j]", which matches the result of
    # `qc_utils.gates.rx(1) @ np.array([[1],[0]])`.
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

    def initialize(
            self, 
            state: str, 
            **kwargs,
        ) -> None:
        """To be implemented by child class. This method should initialize
        self.state to the '0', '1', '+', or 'i' state.
        """
        raise NotImplementedError
    
    def apply_process(self, **kwargs) -> None:
        """To be implemented by child class.
        
        This method should apply the process of interest to self.state.
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
    
    def run_process_tomography(self, **kwargs) -> NDArray[np.complex128]:
        """Requires that methods implementing initialization, logical
        operations, and measurement have been properly defined.
        
        Args:
            kwargs: arguments to pass into user-defined functions.

        Returns:
            The chi matrix characterizing the process.
        """
        if self.num_qubits == 1:
            # following method in Box 8.5 of N&C.
            rhos = []
            state_list = [
                '0', '1', '+', 'i'
            ]
            for state in state_list:
                # perform state tomography for each state created by `prep_ops`.
                def create_state(**kwargs_process):
                    self.initialize(state)
                    self.apply_process(**kwargs)
                state_tomography_helper = StateTomographyHelper(
                    self.StateType,
                    self.num_qubits,
                    create_state,
                    self.measure_X,
                    self.measure_Y,
                    self.measure_Z,
                )

                rho = state_tomography_helper.run_state_tomography(**kwargs)
                rhos.append(rho)
            
            rho1 = rhos[0]
            rho4 = rhos[1]
            rho2 = rhos[2] - 1j*rhos[3] - (1-1j)*(rho1+rho4)/2
            rho3 = rhos[2] + 1j*rhos[3] - (1+1j)*(rho1+rho4)/2

            lambda_mat = 1/2*np.array([
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1,-1, 0],
                [1, 0, 0,-1]
            ], complex)

            rho_mat = np.zeros((4,4), complex)
            rho_mat[:2,:2] = rho1
            rho_mat[2:,:2] = rho3
            rho_mat[:2,2:] = rho2
            rho_mat[2:,2:] = rho4

            chi = np.array(lambda_mat @ rho_mat @ lambda_mat, complex)
            return chi
        else:
            raise NotImplementedError
        
    def chi_matrix_basis_elements(self) -> list[NDArray[np.complex128]]:
        """Return the basis elements E_i that are used together with the chi
        matrix to define a quantum process.
        
        Returns:
            List of basis elements E_i such that 
            `rho' += sum_{m,n}(E_m @ rho @ E_n^dag * chi[m,n])`.
        """
        if self.num_qubits == 1:
            return [
                gates.i2,
                gates.X,
                -1j*gates.Y,
                gates.Z
            ]
        else:
            raise NotImplementedError
        

class GateBasedProcessTomography(ProcessTomography):
    """Identical to ProcessTomography, but user defines gate operations instead
    of multiple reset and measure operations.
    """
    # TODO