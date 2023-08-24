# qc_utils
General utility functions/definitions for quantum computing.

Mostly gate utilities at the moment - defining common gates (X, H, CNOT) and gate constructors (rz, etc).

## Usage

Clone into whatever repository you are running code in.

```
from qc_utils import gates
from qc_utils.kron import nkp
from qc_utils.qctrl import hamiltonians as hams_qctrl
```

## Style guide
I am currently slowly transitioning all docstring comments to Google's standard and adding type annotations to all functions.