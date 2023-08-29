import numpy as np
from numpy.typing import NDArray
from qc_utils.tomography.state_tomography import StateTomography
import typing
import qc_utils.gates as gates

class StateTomographyHelper(StateTomography):
    def __init__(self, qubits, StateType, initialize_fn, H_op_fn, measure_X_fn, measure_Y_fn, measure_Z_fn):
        super().__init__(qubits, StateType)
        self.initialize_fn = initialize_fn
        self.H_op_fn = H_op_fn
        self.measure_X_fn = measure_X_fn
        self.measure_Y_fn = measure_Y_fn
        self.measure_Z_fn = measure_Z_fn
    
    def initialize(self, **kwargs):
        return self.initialize_fn(**kwargs)

    def H_op(self, qubits, **kwargs):
        return self.H_op_fn(qubits, **kwargs)
    
    def measure_X(self, qubits, **kwargs):
        return self.measure_X_fn(qubits, **kwargs)
    
    def measure_Y(self, qubits, **kwargs):
        return self.measure_Y_fn(qubits, **kwargs)
    
    def measure_Z(self, qubits, **kwargs):
        return self.measure_Z_fn(qubits, **kwargs)

class ProcessTomography():
    """Class used to define a general process tomography experiment, without
    assuming anything about the type or behavior of the state representation.
    
    Requires user to define the functions that relate to operating on the 
    system.

    Currently limited to a single logical qubit.
            
    Attributes:
        qubits: the indices of the qubits on which to perform tomography.
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
            super().__init__(0, qiskit.QuantumCircuit)
            self.state: qiskit.QuantumCircuit = qiskit.QuantumCircuit(1, 1)
        
        def initialize(self, **kwargs):
            self.state = qiskit.QuantumCircuit(1, 1)
        
        def apply_process(self, **kwargs):
            self.state.rx(1, 0)

        def X_op(self, tgts, **kwargs):
            self.state.x(tgts)

        def H_op(self, tgts, **kwargs):
            self.state.h(tgts)

        def measure_X(self, tgts, **kwargs):
            self.state.h(tgts)
            return self.measure_Z(tgts)

        def measure_Y(self, tgts, **kwargs):
            self.state.sdg(tgts)
            self.state.h(tgts)
            return self.measure_Z(tgts)

        def measure_Z(self, tgts, **kwargs):
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
    `qc_utils.gates.rx(1) @ np.array([[1],[0]])`.
    ```
    """

    def __init__(self, qubits: int | list[int] = 0, StateType = typing.Any):
        """Initialize the StateTomography object.
        
        Args:
            qubits: list of qubits on which we want to perform state
                tomography. If None, assumes that the system consists of only
                a single qubit.
        """
        self.tgt_qubits = qubits
        if type(qubits) is int:
            self.num_qubits = 1
        else:
            assert type(qubits) is list[int]
            self.num_qubits = len(qubits)
        self.StateType = StateType
        self.state: StateType | None = None

    def initialize(self) -> None:
        """To be implemented by child class.

        This method should initialize self.state to the 0 state.
        """
        raise NotImplementedError
    
    def apply_process(self, **kwargs) -> None:
        """To be implemented by child class.
        
        This method should apply the process of interest to self.state.
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

    def X_op(self, qubits: int | list[int], **kwargs) -> None:
        """To be implemented by child class.

        This method should apply an inplace logical X operation to self.state
        on the qubits specified by `qubits`.

        Args:
            qubits: the qubit indices on which to apply the operation.
        """
        raise NotImplementedError

    def measure_X(self, qubits: int | list[int], **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the X basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError
    
    def measure_Y(self, qubits: int | list[int], **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the Y basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError
    
    def measure_Z(self, qubits: int | list[int], **kwargs) -> list[float]:
        """To be implemented by child class.

        This method should generate measurement results from measuring
        self.state in the Z basis. For a single qubit, the 
        results should be a list of length 2 corresponding to 
        [fraction_measure_0, fraction_measure_1]. The list should always be 
        normalized.
        """
        raise NotImplementedError
    
    def run_process_tomography(self, **kwargs) -> NDArray[np.complex_]:
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
            prep_ops_list = [[], [self.X_op], [self.H_op], [self.X_op, self.H_op]]
            for prep_ops in prep_ops_list:
                # perform state tomography for each state created by `prep_ops`.
                def create_state(**kwargs_process):
                    self.initialize(**kwargs_process)
                    for op in prep_ops:
                        op(self.tgt_qubits, **kwargs_process)
                    self.apply_process(**kwargs)
                state_tomography_helper = StateTomographyHelper(
                    self.tgt_qubits, 
                    self.StateType,
                    create_state,
                    self.H_op,
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
            rho_mat[:2,2:] = rho2
            rho_mat[2:,:2] = rho3
            rho_mat[2:,2:] = rho4

            chi = np.array(lambda_mat @ rho_mat @ lambda_mat, complex)
            return chi
        else:
            raise NotImplementedError
        
    def chi_matrix_basis_elements(self) -> list[NDArray[np.complex_]]:
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