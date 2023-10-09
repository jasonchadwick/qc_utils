import datetime
import numpy as np
from numpy.typing import NDArray

def dt_linspace(
        start_dt: datetime.datetime, 
        end_dt: datetime.datetime, 
        num: int,
        numpy: bool = True,
    ) -> NDArray[np.datetime64] | list[datetime.datetime]:
    """Return evenly-spaced array of datetime objects between start and end
    (inclusive).
    
    Args:
        start_dt: starting datetime.
        end_dt: ending datetime.
        num: number of elements requested.
    
    Returns:
        Numpy array of evenly-spaced datetime objects.
    """
    delta = end_dt - start_dt
    values = [start_dt + datetime.timedelta(0, f*delta.total_seconds()) for f in np.linspace(0,1,num)]
    if numpy:
        return np.array(values, dtype=np.datetime64)
    else:
        return values