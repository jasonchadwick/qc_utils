import numpy as np
from functools import reduce

def idx_from_bits(bits, nt):
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

def bits_from_idx(idx, nt):
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