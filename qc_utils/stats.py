import numpy as np
from numpy.typing import NDArray
import scipy
import math

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

def get_most_probable_bitstrings(biases: NDArray[np.float_], n_bitstrings, probability_threshold=0.0):
    """Find the most probable bitstrings sampled from a set of biased coins.
    Developed with Max Seifert and Maria Vinokurskaya.

    Main idea: the most likely bitstring is simply the most likely value for
    each coin. The nth most likely bitstring is only one coin flip away from one
    of the top (n-1) bitstrings. We can thus find the top n bitstrings by
    iteratively trying all single-bit flips on the previously-found bitstrings.

    Args:
        biases: Coin biases.
        n_bitstrings: Number of bitstrings to return. Capped at 2**len(biases).
        probability_threshold: If nonzero, only return bitstrings with
            probability greater than this value (limited to at most
            n_bitstrings).
    
    Returns:
        bitstrings: (n_bitstrings, len(biases)) Boolean array, where each row is
            a bitstring.
        probabilities: Array of probabilities corresponding to bitstrings.
    """
    n = biases.shape[0]
    n_bitstrings = min(n_bitstrings, 2**n)

    most_probable_bitstring = np.round(biases).astype(bool)

    chosen_bitstrings = np.zeros((n_bitstrings, n), bool)
    chosen_bitstrings[0,:] = most_probable_bitstring
    probabilities = [np.prod(biases * most_probable_bitstring + (1 - biases) * (1 - most_probable_bitstring))]
    for i in range(n_bitstrings-1):
        chosen_bitstring = None
        chosen_prob = 0
        for flip_bit in range(n):
            flip_mask = np.zeros(n, bool)
            flip_mask[flip_bit] = 1

            # flip single bit in all previous bitstrings
            flipped_bitstrings = np.logical_xor(chosen_bitstrings[:i+1], flip_mask)
            
            for flipped_bitstring in flipped_bitstrings:
                # remove bitstrings that we have already seen
                if not np.any(np.all(chosen_bitstrings[:i+1] == flipped_bitstring, axis=1)):
                    flipped_prob = np.prod(biases*flipped_bitstring + (1-biases)*(1-flipped_bitstring))
                    # find highest-probability remaining bitstring
                    if flipped_prob > chosen_prob:
                        chosen_bitstring = flipped_bitstring
                        chosen_prob = flipped_prob
        if chosen_prob < probability_threshold:
            return chosen_bitstrings[:i+1], np.array(probabilities)[:i+1]
        chosen_bitstrings[i+1,:] = chosen_bitstring
        probabilities.append(chosen_prob)
    
    return chosen_bitstrings, np.array(probabilities)

def log_factorial(n: int) -> float:
    """Approximates $\ln(n!)$; the natural logarithm of a factorial. Copied from
    sinter's `log_factorial` function in `sinter/_probability_util.py`.

    Args:
        n: The input to the factorial.

    Returns:
        Evaluates $ln(n!)$ using `math.lgamma(n+1)`.
    """
    return math.lgamma(n + 1)

def log_binomial(*, p: float | np.ndarray, n: int, hits: int) -> np.ndarray:
    """Approximates the natural log of a binomial distribution's probability.
    Copied from sinter's `log_binomial` function in
    `sinter/_probability_util.py`.

    This method evaluates $\ln(P(hits = B(n, p)))$, with all computations done
    in log space to ensure intermediate values can be represented as floating
    point numbers without underflowing to 0 or overflowing to infinity. This
    method can be broadcast over multiple hypothesis probabilities by giving a
    numpy array for `p` instead of a single float.

    Args:
        p: The hypotehsis probability. The independent probability of a hit
            occurring for each sample. This can also be an array of
            probabilities, in which case the function is broadcast over the
            array.
        n: The number of samples that were taken.
        hits: The number of hits that were observed amongst the samples that
            were taken.

    Returns:
        $\ln(P(hits = B(n, p)))$
    """
    # Clamp probabilities into the valid [0, 1] range (in case float error put them outside it).
    p_clipped = np.clip(p, 0, 1)

    result = np.zeros(shape=p_clipped.shape, dtype=np.float32)
    misses = n - hits

    # Handle p=0 and p=1 cases separately, to avoid arithmetic warnings.
    if hits:
        result[p_clipped == 0] = -np.inf
    if misses:
        result[p_clipped == 1] = -np.inf

    # Multiply p**hits and (1-p)**misses onto the total, in log space.
    result[p_clipped != 0] += np.log(p_clipped[p_clipped != 0]) * hits
    result[p_clipped != 1] += np.log1p(-p_clipped[p_clipped != 1]) * misses

    # Multiply (n choose hits) onto the total, in log space.
    log_n_choose_hits = log_factorial(n) - log_factorial(misses) - log_factorial(hits)
    result += log_n_choose_hits

    return result

def fit_binomial(
        *,
        num_shots: int,
        num_hits: int,
        max_likelihood_factor: float) -> tuple[float, float]:
    """Determine hypothesis probabilities compatible with the given hit ratio.
    Modified from sinter's `fit_binomial` function in 
    `sinter/_probability_util.py`.

    The result includes the best fit (the max likelihood hypothis) as well as
    the smallest and largest probabilities whose likelihood is within the given
    factor of the maximum likelihood hypothesis.

    Args:
        num_shots: The number of samples that were taken.
        num_hits: The number of hits that were seen in the samples.
        max_likelihood_factor: The maximum Bayes factor between the low/high
            hypotheses and the best hypothesis (the max likelihood hypothesis).
            This value should be larger than 1 (as opposed to between 0 and 1).

    Returns:
        A `sinter.Fit` with the low, best, and high hypothesis probabilities.
    """
    if max_likelihood_factor < 1:
        raise ValueError(f'max_likelihood_factor={max_likelihood_factor} < 1')
    if num_shots == 0:
        return (0.0, 1.0)
    log_max_likelihood = log_binomial(p=num_hits / num_shots, n=num_shots, hits=num_hits)
    target_log_likelihood = log_max_likelihood - math.log(max_likelihood_factor)
    acc = 100
    low = scipy.optimize.minimize(
        func=lambda exp_err: log_binomial(p=exp_err / (acc * num_shots), n=num_shots, hits=num_hits) - target_log_likelihood,
        x0=num_hits*acc,
        bounds=(0, num_hits * acc),
    ) / acc
    high = scipy.optimize.minimize(
        func=lambda exp_err: -log_binomial(p=exp_err / (acc * num_shots), n=num_shots, hits=num_hits) + target_log_likelihood,
        x0=num_hits*acc,
        bounds=(num_hits * acc, num_shots * acc),
    ) / acc
    return (low / num_shots, high / num_shots)