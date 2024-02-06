import numpy as np
from numpy.typing import NDArray
import scipy

def likelihood_ratio_CI(count: float, ntrials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval on an estimated proportion via Likelihood Ratio Test.
    From https://online.stat.psu.edu/stat504/book/export/html/657.

    Args:
        count: observed number of successes (must be between 0 and ntrials).
        ntrials: total number of trials.
        confidence: desired confidence interval between 0 and 1 (i.e. 0.95 for 95% CI).

    Returns:
        Lower and upper CI bounds on the estimated proportion.
    """
    p_hat = count / ntrials
    if p_hat == 0 or p_hat == 1:
        return p_hat, p_hat
    z = scipy.stats.norm.ppf([1-(1-confidence)/2])
    def gsquared(p):
        return 2*(count*np.log(p_hat / p) + (ntrials - count)*np.log((1 - p_hat) / (1 - p))) - z**2
    lower_bound = scipy.optimize.root_scalar(gsquared, bracket=(1e-10, p_hat)).root
    upper_bound = scipy.optimize.root_scalar(gsquared, bracket=(p_hat, 1-1e-10)).root
    return lower_bound, upper_bound

def confidence_interval(
        x: NDArray[np.float_], 
        xh: NDArray[np.float_], 
        stderr: NDArray[np.float_], 
        confidence: float,
        degrees_of_freedom: int,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Compute upper and lower confidence interval bounds for plotting
    prediction intervals. From
    https://www2.stat.duke.edu/courses/Spring14/sta101.001/Sec9-3.pdf. 
    
    Args:
        x: observed x values.
        xh: x values to compute prediction interval at.
        stderr: standard error of actual y values from predicted.
        confidence: desired confidence (e.g. 0.95).
        degrees_of_freedom: len(x) - len(fit_params).
    
    Returns:
        Confidence interval bound. We are confident that the mean is within the
        range (yh-bound, yh+bound).
    """
    n = len(x)
    t_val = scipy.stats.t.interval(confidence, degrees_of_freedom)[1] / scipy.stats.t.std(degrees_of_freedom)
    return t_val*stderr*np.sqrt(1/n + (xh-np.mean(x))**2/((n-1)*np.var(x)))

def prediction_interval(
        x: NDArray[np.float_], 
        xh: NDArray[np.float_], 
        stderr: NDArray[np.float_], 
        confidence: float,
        degrees_of_freedom: int,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Compute upper and lower prediction interval bounds for plotting
    prediction intervals. From
    https://www2.stat.duke.edu/courses/Spring14/sta101.001/Sec9-3.pdf. 
    
    Args:
        x: observed x values.
        xh: x values to compute prediction interval at.
        yh: predicted y values (from fitted function).
        stderr: standard error of actual y values from predicted.
        confidence: desired confidence (e.g. 0.95).
        degrees_of_freedom: len(x) - len(fit_params).
    
    Returns:
        Confidence interval bound. We are confident that a prediction yh is
        within the range (yh-bound, yh+bound).
    """
    n = len(x)
    t_val = scipy.stats.t.interval(confidence, degrees_of_freedom)[1] / scipy.stats.t.std(degrees_of_freedom)
    return t_val*stderr*np.sqrt(1 + 1/n + (xh-np.mean(x))**2/((n-1)*np.var(x)))

def get_most_probable_bitstrings(biases: NDArray[np.float_], n_bitstrings):
    """Find the most probable bitstrings sampled from a set of biased coins.
    Developed with Max Seifert and Maria Vinokurskaya.

    Main idea: the most likely bitstring is simply the most likely value for
    each coin. The nth most likely bitstring is only one coin flip away from one
    of the top (n-1) bitstrings. We can thus find the top n bitstrings by
    iteratively trying all single-bit flips on the previously-found bitstrings.

    Args:
        biases: Coin biases.
        n_bitstrings: Number of bitstrings to return. Capped at 2**len(biases).
    
    Returns:
        bitstrings: (n_bitstrings, len(biases)) Boolean array, where each row is
            a bitstring.
        probabilities: Array of probabilities corresponding to bitstrings.
    """
    n = biases.shape[0]
    n_bitstrings = min(n_bitstrings, 2**n)

    most_probable_bitstring = np.round(biases).astype(bool)

    # turn into fixed-size array
    chosen_bitstrings = np.zeros((n_bitstrings, n), bool)
    chosen_bitstrings[0,:] = most_probable_bitstring
    probabilities = [np.prod(biases * most_probable_bitstring + (1 - biases) * (1 - most_probable_bitstring))]
    last_prob = 1.0
    for i in range(n_bitstrings-1):
        chosen_bitstring = None
        chosen_prob = 0
        for flip_bit in range(n):
            flip_mask = np.zeros(n, bool)
            flip_mask[flip_bit] = 1
            # modify all previous bitstrings
            flipped_bitstrings = np.logical_xor(chosen_bitstrings[:i+1], flip_mask)

            # remove bitstrings that we have already seen
            for flipped_bitstring in flipped_bitstrings:
                # array logic to see if any chosen bitstrings overlap
                if not np.any(np.all(chosen_bitstrings[:i+1] == flipped_bitstring, axis=1)):
                    flipped_prob = np.prod(biases * flipped_bitstring + (1 - biases) * (1 - flipped_bitstring))
                    if flipped_prob > chosen_prob:
                        chosen_bitstring = flipped_bitstring
                        chosen_prob = flipped_prob
        last_prob = chosen_prob
        chosen_bitstrings[i+1,:] = chosen_bitstring
        probabilities.append(chosen_prob)
    
    return chosen_bitstrings, np.array(probabilities)