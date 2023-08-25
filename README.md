# qc_utils
General utility functions/definitions for quantum computing. Includes a wide variety of things such as defining numpy arrays of commonly-used gates, working with density matrices, generating Hamiltonians, and even performing state and process tomography.
## Usage

Clone into whatever repository you are running code in.

```
from qc_utils import gates
from qc_utils.kron import nkp
from qc_utils.qctrl import hamiltonians as hams_qctrl
```

## Style guide
I generally try to adhere to Google's Python style guide (mainly for function docstrings), with 80-char line limit.