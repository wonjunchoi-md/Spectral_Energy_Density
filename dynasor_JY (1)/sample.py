import numpy as np
import pandas as pd
from typing import Dict, Any


class Sample:
    """
    Class for holding correlation functions and some additional meta data.
    Sample objects can be written to and read from file.

    Parameters
    ----------
    data_dict
        Dictionary with correlation functions.
    meta_data
        Dictionary with meta data, for example atom-types, simulation cell, number of atoms,
        time stamps, user names, etc.
    """

    def __init__(self, data_dict: Dict[str, Any], **meta_data: Dict[str, Any]):

        # set data dict as attributes
        self._data_keys = list(data_dict)
        for key in data_dict:
            setattr(self, key, data_dict[key])

        # set system parameters
        self.meta_data = meta_data
        self._atom_types = meta_data['atom_types']
        self._pairs = meta_data['pairs']
        self._particle_counts = meta_data['particle_counts']
        self._cell = meta_data['cell']

    def __getitem__(self, key):
        """ Makes it possible to get the attributes using Sample['key'] """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def write_to_npz(self, fname: str):
        """ Write object to file in numpy's npz format.

        Parameters
        ----------
        fname
            Name of the file in which to store the Sample object.
        """
        data_to_save = dict(name=self.__class__.__name__)
        data_to_save['meta_data'] = self.meta_data
        data_dict = dict()
        for key in self._data_keys:
            data_dict[key] = getattr(self, key)
        data_to_save['data_dict'] = data_dict
        np.savez_compressed(fname, **data_to_save)

    @property
    def available_correlation_functions(self):
        """ All the available correlation functions in sample. """
        keys_to_skip = set(['q_points', 'q_norms', 'time', 'omega'])
        return sorted(list(set(self._data_keys) - keys_to_skip))

    @property
    def dimensions(self):
        r"""The dimensions for the samples, e.g., for :math:`S(q, \omega)`
        the dimensions would be the :math:`q` and :math:`\omega` axes.
        """
        keys_to_skip = set(self.available_correlation_functions)
        return sorted(list(set(self._data_keys) - keys_to_skip))

    @property
    def atom_types(self):
        if self._atom_types is None:
            return None
        return self._atom_types.copy()

    @property
    def particle_counts(self):
        if self._particle_counts is None:
            return None
        return self._particle_counts.copy()

    @property
    def pairs(self):
        if self._pairs is None:
            return None
        return self._pairs.copy()

    @property
    def cell(self):
        if self._cell is None:
            return None
        return self._cell.copy()

    def __repr__(self):
        return str(self)

    def __str__(self):
        s_contents = [self.__class__.__name__]
        s_contents.append(f'Atom types: {self.atom_types}')
        s_contents.append(f'Pairs: {self.pairs}')
        s_contents.append(f'Particle counts: {self.particle_counts}')
        s_contents.append('Simulations cell:')
        s_contents.append(f'{self.cell}')
        for key in self.dimensions:
            s_i = f'{key:15} with shape: {np.shape(getattr(self, key))}'
            s_contents.append(s_i)
        for key in self.available_correlation_functions:
            s_i = f'{key:15} with shape: {np.shape(getattr(self, key))}'
            s_contents.append(s_i)
        s = '\n'.join(s_contents)
        return s

    def _repr_html_(self) -> str:
        s = [f'<h3>{self.__class__.__name__}</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left">Field</th>'
              '<th>Content/Size</th></tr></thead>']
        s += ['<tbody>']
        s += ['<tr><td style="text-align: left">Atom types</td>'
              f'<td>{self.atom_types}</td></tr>']
        s += ['<tr><td style="text-align: left">Pairs</td>'
              f'<td>{self.pairs}</td></tr>']
        s += ['<tr><td style="text-align: left">Particle counts</td>'
              f'<td>{self.particle_counts}</td></tr>']
        s += ['<tr><td style="text-align: left">Simulations cell</td>'
              f'<td>{self.cell}</td></tr>']
        for key in self._data_keys:
            s += [f'<tr><td style="text-align: left">{key}</td>'
                  f'<td>{np.shape(getattr(self, key))}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return '\n'.join(s)

    @property
    def has_incoherent(self):
        """ Whether this sample contains the incoherent correlation functions or not. """
        return False

    @property
    def has_currents(self):
        """ Whether this sample contains the current correlation functions or not. """
        return False


class StaticSample(Sample):

    def to_dataframe(self):
        """ Returns correlation functions as pandas dataframe """
        df = pd.DataFrame()
        for dim in self.dimensions:
            df[dim] = self[dim].tolist()  # to list to make q-points (N, 3) work in dataframe
        for key in self.available_correlation_functions:
            df[key] = self[key].reshape(-1, )
        return df


class DynamicSample(Sample):

    @property
    def has_incoherent(self):
        return 'Fqt_incoh' in self.available_correlation_functions

    @property
    def has_currents(self):
        pair_string = '_'.join(self.pairs[0])
        return f'Clqt_{pair_string}' in self.available_correlation_functions

    def to_dataframe(self, q_index: int):
        """ Returns correlation functions as pandas dataframe for the given q-index.

        Parameters
        ----------
        q_index
            index of q-point to return
        """
        df = pd.DataFrame()
        for dim in self.dimensions:
            if dim in ['q_points', 'q_norms']:
                continue
            df[dim] = self[dim]
        for key in self.available_correlation_functions:
            df[key] = self[key][q_index]
        return df


def read_sample_from_npz(fname: str) -> Sample:
    """ Read :class:`Sample <dynasor.sample.Sample>` from file.

    Parameters
    ----------
    fname
        Path to the file (numpy npz format) from which to read
        the :class:`Sample <dynasor.sample.Sample>` object.
    """
    data_read = np.load(fname, allow_pickle=True)
    data_dict = data_read['data_dict'].item()
    meta_data = data_read['meta_data'].item()
    if data_read['name'] == 'StaticSample':
        return StaticSample(data_dict, **meta_data)
    elif data_read['name'] == 'DynamicSample':
        return DynamicSample(data_dict, **meta_data)
    else:
        return Sample(data_dict, **meta_data)
