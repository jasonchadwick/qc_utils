import numpy as np
from functools import reduce

def idx_from_bits(bits, nt):
    nbits = len(nt)
    acc = 0
    for b in range(nbits):
        bit_value = reduce(lambda x,y:x*y, nt[b+1:], 1)
        acc += bits[b]*bit_value
    return acc

def bits_from_idx(idx, nt):
    nbits = len(nt)
    acc = []
    for b in range(nbits):
        bit_value = reduce(lambda x,y:x*y, nt[b+1:], 1)
        acc.append(idx // bit_value)
        idx %= bit_value
    return acc