"""
A number of utility functions, for example for dealing with
autocorrelation functions, Fourier transforms, and smoothing.
"""

import numpy as np
from scipy.signal import correlate
from numpy.typing import NDArray
import pandas as pd


def compute_acf(Z: NDArray[float], delta_t: float = 1.0, method='scipy'):
    r"""
    Computes the autocorrelation function (ACF) for a one-dimensional signal :math:`Z` in time as

    .. math::

        ACF(\tau) = \frac{\left < Z(t) Z^*(t+\tau) \right >}{\left <  Z(t)  Z^*(t) \right >}

    Here, only the real part of the ACF is returned since if :math:`Z` is complex
    the imaginary part should average out to zero for any stationary signal.

    Parameters
    ----------
    Z
        complex time signal
    delta_t
        spacing in time between two consecutive values in :math:`Z`
    method
        implementation to use; possible values: `numpy` and `scipy` (default and usually faster)
    """

    # keep only real part and normalize
    acf = _compute_correlation_function(Z, Z, method)
    acf = np.real(acf)
    acf /= acf[0]

    time_lags = delta_t * np.arange(0, len(acf), 1)
    return time_lags, acf


def _compute_correlation_function(Z1, Z2, method='scipy'):
    N = len(Z1)
    assert len(Z1) == len(Z2)
    if method == 'scipy':
        cf = correlate(Z1, Z2, mode='full')[N - 1:] / np.arange(N, 0, -1)
    elif method == 'numpy':
        cf = np.correlate(Z1, Z2, mode='full')[N - 1:] / np.arange(N, 0, -1)
    else:
        raise ValueError('method must be either numpy or scipy')
    return cf


# smoothing functions / FFT filters
# -------------------------------------
def gaussian_decay(t: NDArray[float], t_sigma: float):
    r"""
    Evaluates a gaussian distribution in time :math:`f(t)`, which can be applied to an ACF in time
    to artificially damp it, i.e., forcing it to go to zero for long times.

    .. math::

        f(t) = \exp{\left [-\frac{1}{2} \left (\frac{t}{t_\mathrm{sigma}}\right )^2 \right ] }

    Parameters
    ----------
    t
        time array
    t_sigma
        width (standard deviation of the gaussian) of the decay
    """

    return np.exp(- 1 / 2 * (t / t_sigma) ** 2)


def fermi_dirac(t: NDArray[float], t_0: float, t_width: float):
    r"""
    Evaluates a Fermi-Dirac-like function in time :math:`f(t)`, which can be applied to an ACF in
    time to artificially damp it, i.e., forcing it to go to zero for long times without affecting
    the short-time correlations too much.

    .. math::

        f(t) = \frac{1}{\exp{[(t-t_0)/t_\mathrm{width}}] + 1}

    Parameters
    ----------
    t
        time array
    t_0
        starting time for decay
    t_width
        width of the decay

    """
    return 1.0 / (np.exp((t - t_0) / t_width) + 1)


def smoothing_function(data: NDArray[float], window_size: int, window_type: str = 'hamming'):
    """
    Smoothing function for 1D arrays.
    This functions employs pandas rolling window average

    Parameters
    ----------
    data
        1D data array
    window_size
        The size of smoothing/smearing window
    window_type
        What type of window-shape to use, e.g. ``'blackman'``, ``'hamming'``, ``'boxcar'``
        (see pandas and scipy documentaiton for more details)

    """
    series = pd.Series(data)
    new_data = series.rolling(window_size, win_type=window_type, center=True, min_periods=1).mean()
    return np.array(new_data)
