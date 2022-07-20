import numpy as np

def idx_from_bits(bits, endian='big'):
    acc = 0
    for i,b in enumerate(bits):
        if endian == 'big':
            acc += b*2**(len(bits)-i-1)
        else:
            acc += b*2**i
    return acc

def bits_from_idx(idx, nt, endian='big'):
    acc = []
    for place in range(nt):
        if endian == 'big':
            value = 2**(nt-place-1)
        else:
            value = 2**place
        acc.append(idx // value)
        idx %= value
    return acc