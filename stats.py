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
    z = scipy.stats.norm.ppf([1-(1-confidence)/2])
    def gsquared(p):
        return 2*(count*np.log(p_hat / p) + (ntrials - count)*np.log((1 - p_hat) / (1 - p))) - z**2
    lower_bound = scipy.optimize.root_scalar(gsquared, bracket=(1e-5, p_hat)).root
    upper_bound = scipy.optimize.root_scalar(gsquared, bracket=(p_hat, 1-1e-5)).root
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