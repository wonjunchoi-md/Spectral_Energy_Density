from typing import Dict


class Weights:
    """
    Class holding weights and support functions for weighting of samples

    Parameters
    ----------
    weights_coh
        A dict with keys and values representing the atom types and their corresponding
        coherent scattering length, ``{'A': b_A }``.
    weights_incoh
        A dict with keys and values representing the atom types and their corresponding
        incoherent scattering length, ``{'A': b_A }``.
    supports_currents
        whether or not the coherent weights should be applied to current-correlation functions
    """

    def __init__(
        self,
        weights_coh: Dict[str, float],
        weights_incoh: Dict[str, float] = None,
        supports_currents: bool = True,
    ):
        self._weights_coh = weights_coh
        self._weights_incoh = weights_incoh
        self._supports_currents = supports_currents

    def get_weight_coh(self, atom_type, q_norm=None):
        """Get the coherent weight for a given atom type and q-vector norm."""
        return self._weights_coh[atom_type]

    def get_weight_incoh(self, atom_type, q_norm=None):
        """Get the incoherent weight for a given atom type and q-vector norm."""
        if self._weights_incoh is None:
            return None
        return self._weights_incoh[atom_type]

    @property
    def supports_currents(self):
        """
        Wether or not this :class:`Weights` object supports weighting of current correlations.
        """
        return self._supports_currents

    @property
    def supports_incoherent(self):
        """
        Whether or not this :class:`Weights` object supports weighting of incoherent
        correlation functions.
        """
        return self._weights_incoh is not None

    def __str__(self):
        s = ['weights coherent:']
        for key, val in self._weights_coh.items():
            s.append(f'  {key}: {val}')

        # Return early if incoherent weights
        # are None
        if self._weights_incoh is None:
            return '\n'.join(s)

        s.append('weights incoherent:')
        for key, val in self._weights_incoh.items():
            s.append(f'  {key}: {val}')
        return '\n'.join(s)
