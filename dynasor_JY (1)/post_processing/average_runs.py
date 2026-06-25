import numpy as np
from dynasor.sample import Sample
from typing import List
from copy import deepcopy


def get_sample_averaged_over_independent_runs(
        samples: List[Sample], live_dangerously=False) -> Sample:
    """
    Compute an averaged sample from multiple samples obtained from identical independent runs.

    Note, all the meta_data and dimensions in all samples must be the same,
    else ValueError is raised (unless ``live_dangerously`` is set to True).

    Parameters
    ----------
    samples
        list of all sample objects to be averaged over
    live_dangerously
        setting True allows for averaging over samples which meta-data information is not identical.
    """

    # get meta data and dimensions from first sample
    sample_ref = samples[0]
    data_dict = dict()
    meta_data = deepcopy(sample_ref.meta_data)

    # test that all samples have identical dimensions
    for sample in samples:
        assert sorted(sample.dimensions) == sorted(sample_ref.dimensions)
        for dim in sample_ref.dimensions:
            assert np.allclose(sample[dim], sample_ref[dim])

    for dim in sample_ref.dimensions:
        data_dict[dim] = sample_ref[dim]

    # test that all samples have identical meta_data
    if not live_dangerously:
        for sample in samples:
            assert len(sample.meta_data) == len(meta_data)

            for key, val in meta_data.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        assert sample_ref.meta_data[key][k] == sample.meta_data[key][k]
                elif isinstance(val, np.ndarray):
                    assert np.allclose(sample.meta_data[key], val)
                elif isinstance(val, float):
                    assert np.isclose(sample.meta_data[key], val)
                else:
                    assert sample.meta_data[key] == val

    # average all correlation functions
    for key in sample.available_correlation_functions:
        data = []
        for sample in samples:
            data.append(sample[key])
        data_average = np.nanmean(data, axis=0)
        data_dict[key] = data_average

    return sample.__class__(data_dict, **meta_data)
