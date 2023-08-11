import numpy as np
import scipy

def likelihood_ratio_CI(count: float, ntrials: int, confidence: float = 0.95) -> tuple(float, float):
    """Compute confidence interval on an estimated proportion via Likelihood Ratio Test.
    From https://online.stat.psu.edu/stat504/book/export/html/657.

    Args:
        count: observed number of successes.
        ntrials: total number of trials.
        confidence: desired confidence interval (i.e. 0.95 for 95% CI).

    Returns:
        Lower and upper CI bounds on the estimated proportion.
    """
    p_hat = count / ntrials
    z = scipy.stats.norm.ppf([1-(1-confidence)/2])
    def gsquared(p):
        return 2*(count*np.log(p_hat / p) + (ntrials - count)*np.log((1 - p_hat) / (1 - p))) - z**2
    lower_bound = scipy.optimize.root_scalar(gsquared, bracket=(1e-5, p_hat)).root
    upper_bound = scipy.optimize.root_scalar(gsquared, bracket=(p_hat, 1-1e-5)).root
    return lower_bound, upper_bound