# qc_utils
General utility functions/definitions for quantum computing. Includes a wide
variety of things such as defining numpy arrays of commonly-used gates, working
with density matrices, generating Hamiltonians, and even performing state and
process tomography.

## Installation

Install via pip or poetry or whatever you want.

`poetry add git+https://github.com/jasonchadwick/qc_utils`

`pip install git+https://github.com/jasonchadwick/qc_utils`

## Requirements for full functionality
```
numpy
matplotlib
qiskit
qutip
scipy
sympy
qiskit-ibm-provider
```


## Style guide
I generally try to adhere to Google's Python style guide (mainly for function docstrings), with 80-char line limit.