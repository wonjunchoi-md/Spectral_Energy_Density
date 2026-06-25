import numpy as np

from copy import deepcopy
from dynasor.logging_tools import logger
from dynasor.sample import Sample
from numpy.typing import NDArray
from scipy.stats import norm


def get_spherically_averaged_sample_smearing(
        sample: Sample, q_norms: NDArray[float], q_width: float) -> Sample:
    r"""
    Compute a spherical average over q-points for all the correlation functions in :attr:`sample`.

    In the gaussian average method each q-point contributes to the function value at
    given :math:`\vec{q}` with a weight determined by a gaussian function. For example

    .. math::

        F(q) = \sum_i w(\boldsymbol{q}_i, q) F(\boldsymbol{q}_i)

    where

    .. math::

        w(\boldsymbol{q}_i, q) \propto \exp{\left [ -\frac{1}{2} \left ( \frac{|\boldsymbol{q}_i|
        - q}{q_{width}} \right)^2 \right ]}

    and

    .. math::

        \sum_i w(\boldsymbol{q}_i, q) = 1.0

    This corresponds to a gaussian smearing or convolution.
    The input parameters are :attr:`q_norms`, setting to the values of :math:`|\vec{q}|`,
    for which the function is evaluated and :attr:`q_width` specifying the
    standard deviation of the gaussian smearing.

    Parameters
    ----------
    sample
        Input sample.
    q_norms
        Values of :math:`|\vec{q}|` at which to evaluate the correlation functions.
    q_width
        Standard deviation of the gaussian smearing.
    """
    if not isinstance(sample, Sample):
        raise ValueError('Input sample is not a Sample object.')

    # get q-points
    q_points = sample.q_points
    if q_points.shape[1] != 3:
        raise ValueError('q-points array has the wrong shape.')

    # setup new input dicts for new Sample, remove q_points, add q_norms
    meta_data = deepcopy(sample.meta_data)
    data_dict = dict()
    for key in sample.dimensions:
        if key == 'q_points':
            continue
        data_dict[key] = sample[key]

    for key in sample.available_correlation_functions:
        Z = getattr(sample, key)
        averaged_data = _get_gaussian_average(q_points, Z, q_norms, q_width)
        data_dict[key] = averaged_data
    data_dict['q_norms'] = q_norms

    return sample.__class__(data_dict, **meta_data)


def get_spherically_averaged_sample_binned(sample: Sample, num_q_bins: int) -> Sample:
    r"""
    Compute a spherical average over q-points for all the correlation functions in `:attr:`sample`.

    Here, a q-binning method is used to conduct the spherical average, meaning all q-points are
    placed into spherical bins (shells).
    The corresponding function is calculated as the average of all q-points in a bin.
    If a q-bin does not contain any q-points, then its value is set to ``np.nan``.
    The q_min and q_max are determined from min/max of ``|q_points|``, and will determine
    the q-bin range.
    These will be set as bin-centers for the first and last bins repsectivley.
    The input parameter is the number of q-bins to use :attr:`num_q_bins`.

    Parameters
    ----------
    sample
        Input sample
    num_q_bins
        number of q-bins to use
    """

    if not isinstance(sample, Sample):
        raise ValueError('input sample is not a Sample object.')

    # get q-points
    q_points = sample.q_points
    if q_points.shape[1] != 3:
        raise ValueError('q-points array has wrong shape.')

    # setup new input dicts for new Sample, remove q_points, add q_norms
    meta_data = deepcopy(sample.meta_data)
    data_dict = dict()
    for key in sample.dimensions:
        if key == 'q_points':
            continue
        data_dict[key] = sample[key]

    # compute spherical average for each correlation function
    for key in sample.available_correlation_functions:
        Z = getattr(sample, key)
        q_bincenters, bin_counts, averaged_data = _get_bin_average(q_points, Z, num_q_bins)
        data_dict[key] = averaged_data
    data_dict['q_norms'] = q_bincenters

    return sample.__class__(data_dict, **meta_data)


def _get_gaussian_average(
        q_points: np.ndarray, Z: np.ndarray, q_norms: np.ndarray, q_width: float):

    q_norms_sample = np.linalg.norm(q_points, axis=1)
    Z_average = []
    for q in q_norms:
        weights = _gaussian(q_norms_sample, x0=q, sigma=q_width).reshape(-1, 1)
        norm = np.sum(weights)
        if norm != 0:
            weights = weights / norm
        Z_average.append(np.sum(weights * Z, axis=0))
    return np.array(Z_average)


def _gaussian(x, x0, sigma):
    dist = norm(loc=x0, scale=sigma)
    return dist.pdf(x)


def _get_bin_average(q_points: np.ndarray, data: np.ndarray, num_q_bins: int):
    """
    Compute a spherical average over q-points for the data using q-bins.

    If a q-bin does not contain any q-points, then a np.nan is inserted.

    The q_min and q_min are determined from min/max of |q_points|, and will determine the bin-range.
    These will set as bin-centers for the first and last bins repsectivley.

    Parameters
    ----------
    q_points
        array of q-points shape ``(Nq, 3)``
    data
        data-array of shape ``(Nq, N)``, shape cannot be ``(Nq, )``
    num_q_bins
        number of radial q-point bins to use

    Returns
    -------
    q
        array of |q| bins of shape ``(num_q_bins, )``
    data_averaged
        averaged data-array of shape ``
    """
    N_qpoints = q_points.shape[0]
    N_t = data.shape[1]
    assert q_points.shape[1] == 3
    assert data.shape[0] == N_qpoints

    # q-norms
    q_norms = np.linalg.norm(q_points, axis=1)
    assert q_norms.shape == (N_qpoints,)

    # setup bins
    q_max = np.max(q_norms)
    q_min = np.min(q_norms)
    delta_x = (q_max - q_min) / (num_q_bins - 1)
    q_range = (q_min - delta_x / 2, q_max + delta_x / 2)
    bin_counts, edges = np.histogram(q_norms, bins=num_q_bins, range=q_range)
    q_bincenters = 0.5 * (edges[1:] + edges[:-1])

    # calculate average for each bin
    averaged_data = np.zeros((num_q_bins, N_t))
    for bin_index in range(num_q_bins):
        # find q-indices that belong to this bin
        bin_min = edges[bin_index]
        bin_max = edges[bin_index + 1]
        bin_count = bin_counts[bin_index]
        q_indices = np.where(np.logical_and(q_norms >= bin_min, q_norms < bin_max))[0]
        assert len(q_indices) == bin_count
        logger.debug(f'bin {bin_index} contains {bin_count} q-points')

        # average over q-indices, if no indices then np.nan
        if bin_count == 0:
            logger.warning(f'No q-points for bin {bin_index}')
            data_bin = np.array([np.nan for _ in range(N_t)])
        else:
            data_bin = data[q_indices, :].mean(axis=0)
        averaged_data[bin_index, :] = data_bin

    return q_bincenters, bin_counts, averaged_data
