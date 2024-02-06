import sys
sys.path.append('./')
import numpy as np
import itertools
import qc_utils.stats
from qc_utils.idx import idx_from_bits

# test get_most_probable_bitstrings
print('TESTING get_most_probable_bitstrings...')
n_coins = 10
n_bitstrings = 7

for trial in range(100):
    biases = np.random.rand(n_coins)
    bitstring_probabilities = np.zeros(2**n_coins)
    for i, bitstring in enumerate(itertools.product([0, 1], repeat=n_coins)):
        bitstring_probabilities[i] = np.prod([biases[j] if bitstring[j] == 1 else 1 - biases[j] for j in range(n_coins)])

    bitstrings_sorted = np.argsort(bitstring_probabilities)
    probabilities_sorted = bitstring_probabilities[bitstrings_sorted]

    most_probable_bitstring = np.round(biases).astype(bool)

    chosen_bitstrings, probs = qc_utils.stats.get_most_probable_bitstrings(biases, n_bitstrings)
    chosen_indices = [idx_from_bits(bitstring, [2]*n_coins) for bitstring in chosen_bitstrings]
    assert np.all(np.array(chosen_indices) == bitstrings_sorted[::-1][:n_bitstrings])
    assert np.all(np.isclose(probs, probabilities_sorted[::-1][:n_bitstrings]))
print('PASSED\n')