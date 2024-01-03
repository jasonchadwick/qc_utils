import numpy as np
from functools import reduce
from numpy.typing import NDArray

def idx_from_bits(
        bits: list[int] | NDArray[np.int_], 
        nt: list[int] | NDArray[np.int_]
    ) -> int:
    """Given a list of bit values and a list of corresponding bases, return
    the big-Endian value of those bits.
    
    Example:
        `idx_from_bits([1,1], [2,3]) => 4`

    Args:
        bits: list of bit values.
        nt: list of base values (number of possible bit values in each
            position).
    
    Returns:
        The value of the bits, assuming big-Endian convention.
    """
    bits = np.array(bits)
    nt = np.array(nt)
    if len(bits.shape) == 0:
        bits = bits[None]
    if len(nt.shape) == 0:
        nt = nt[None]
    nbits = nt.shape[0]
    acc = 0
    for b in range(nbits):
        bit_value = reduce(lambda x,y:x*y, nt[b+1:], 1)
        acc += bits[b]*bit_value
    return acc

def bits_from_idx(
        idx: int, 
        nt: list[int] | NDArray[np.int_]
    ) -> NDArray[np.int_]:
    """Given a value and a list of bases, return the big-Endian list of the
    corresponding bits that represent that value.

    Example:
        `bits_from_idx(4, [2,3]) => [1,1]`
    
    Args:
        idx: value to express in bits.
        nt: list of base values (number of possible bit values in each
            position).
    
    Returns:
        The bit values, assuming big-Endian convention.
    """
    nt = np.array(nt)
    if len(nt.shape) == 0:
        nt = nt[None]
    nbits = nt.shape[0]
    acc = []
    for b in range(nbits):
        bit_value = reduce(lambda x,y:x*y, nt[b+1:], 1)
        acc.append(idx // bit_value)
        idx %= bit_value
    return np.array(acc)