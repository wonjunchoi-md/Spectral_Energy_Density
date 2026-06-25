import numpy as np
from typing import Dict
from warnings import warn
from dynasor.post_processing.weights import Weights
from dynasor.sample import Sample, StaticSample, DynamicSample
from copy import deepcopy


def get_weighted_sample(sample: Sample, weights: Weights) -> Sample:
    r"""
    Weights correlation functions with atomic weighting factors

    The weighting of a partial dynamic structure factor
    :math:`S_\mathrm{AB}(\boldsymbol{q}, \omega)`
    for atom types :math:`A` and :math:`B` is carried out as

    .. math::

        S_\mathrm{AB}(\boldsymbol{q}, \omega)
        = f_\mathrm{A}(\boldsymbol{q}) f_\mathrm{B}(\boldsymbol{q})
        S_\mathrm{AB}(\boldsymbol{q}, \omega)

    :math:`f_\mathrm{A}(\boldsymbol{q})` and :math:`f_\mathrm{B}(\boldsymbol{q})`
    are atom-type and :math:`\boldsymbol{q}`-point dependent weights.

    If sample has incoherent correlation functions, but :attr:`weights` does not contain
    information on how to weight the incoherent part, then it will be dropped from the returned
    :attr:`Sample` object (and analogously for current correlation functions).

    Parameters
    ----------
    sample
        input sample to be weighted
    weights
        object containing the weights :math:`f_\mathrm{X}(\boldsymbol{q})`

    Returns
    -------
        A :class:`Sample` instance with the weighted partial and total structure factors.
    """

    # check input arguments
    if sample.has_incoherent and not weights.supports_incoherent:
        warn('The Weights does not support incoherent scattering, dropping the latter '
             'from the weighted sample.')

    if sample.has_currents and not weights.supports_currents:
        warn('The Weights does not support current correlations, dropping the latter '
             'from the weighted sample.')

    # setup new input dicts for new Sample
    data_dict = dict()
    for key in sample.dimensions:
        data_dict[key] = sample[key]
    meta_data = deepcopy(sample.meta_data)

    # generate atomic weights for each q-point and compile to arrays
    if 'q_norms' in sample.dimensions:
        q_norms = sample.q_norms
    else:
        q_norms = np.linalg.norm(sample.q_points, axis=1)

    weights_coh = dict()
    for at in sample.atom_types:
        weight_array = np.reshape([weights.get_weight_coh(at, q) for q in q_norms], (-1, 1))
        weights_coh[at] = weight_array
    if sample.has_incoherent and weights.supports_incoherent:
        weights_incoh = dict()
        for at in sample.atom_types:
            weight_array = np.reshape([weights.get_weight_incoh(at, q) for q in q_norms], (-1, 1))
            weights_incoh[at] = weight_array

    # weighting of correlation functions
    if isinstance(sample, StaticSample):
        data_dict_Sq = _compute_weighting_coherent(sample, 'Sq', weights_coh)
        data_dict.update(data_dict_Sq)
    elif isinstance(sample, DynamicSample):
        # coherent
        Fqt_coh_dict = _compute_weighting_coherent(sample, 'Fqt_coh', weights_coh)
        data_dict.update(Fqt_coh_dict)
        Sqw_coh_dict = _compute_weighting_coherent(sample, 'Sqw_coh', weights_coh)
        data_dict.update(Sqw_coh_dict)

        # incoherent
        if sample.has_incoherent and weights.supports_incoherent:
            Fqt_incoh_dict = _compute_weighting_incoherent(sample, 'Fqt_incoh', weights_incoh)
            data_dict.update(Fqt_incoh_dict)
            Sqw_incoh_dict = _compute_weighting_incoherent(sample, 'Sqw_incoh', weights_incoh)
            data_dict.update(Sqw_incoh_dict)
            data_dict['Fqt'] = data_dict['Fqt_coh'] + data_dict['Fqt_incoh']
            data_dict['Sqw'] = data_dict['Sqw_coh'] + data_dict['Sqw_incoh']
        else:
            data_dict['Fqt'] = data_dict['Fqt_coh'].copy()
            data_dict['Sqw'] = data_dict['Sqw_coh'].copy()

        # currents
        if sample.has_currents and weights.supports_currents:
            Clqt_dict = _compute_weighting_coherent(sample, 'Clqt', weights_coh)
            data_dict.update(Clqt_dict)
            Clqw_dict = _compute_weighting_coherent(sample, 'Clqw', weights_coh)
            data_dict.update(Clqw_dict)

            Ctqt_dict = _compute_weighting_coherent(sample, 'Ctqt', weights_coh)
            data_dict.update(Ctqt_dict)
            Ctqw_dict = _compute_weighting_coherent(sample, 'Ctqw', weights_coh)
            data_dict.update(Ctqw_dict)

    return sample.__class__(data_dict, **meta_data)


def _compute_weighting_coherent(sample: Sample, name: str, weight_dict: Dict):
    """
    Helper function for weighting and summing partial coherent correlation functions.
    """
    data_dict = dict()
    total = np.zeros(sample[name].shape)
    for s1, s2 in sample.pairs:
        key_pair = f'{name}_{s1}_{s2}'
        partial = weight_dict[s1] * weight_dict[s2] * sample[key_pair]
        data_dict[key_pair] = partial
        total += partial
    data_dict[name] = total
    return data_dict


def _compute_weighting_incoherent(sample: Sample, name: str, weight_dict: Dict):
    """
    Helper function for weighting and summing partial incoherent correlation functions.
    """
    data_dict = dict()
    total = np.zeros(sample[name].shape)
    for s1 in sample.atom_types:
        key = f'{name}_{s1}'
        partial = weight_dict[s1] * sample[key]
        data_dict[key] = partial
        total += partial
    data_dict[name] = total
    return data_dict
