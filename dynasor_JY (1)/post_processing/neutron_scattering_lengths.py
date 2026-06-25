import json
from importlib.resources import files
from typing import Dict, List

import numpy as np
from pandas import DataFrame
from .weights import Weights


class NeutronScatteringLengths(Weights):
    """This class provides sample weights corresponding to neutron scattering lengths.
    By default, the coherent and incoherent scattering lengths are weighted by the natural
    abundance of each isotope of the considered atomic species.
    This weighting can be overwritten using the :attr:`abundances` argument.

    The scattering lengths have been extracted from `this NIST
    database <https://www.ncnr.nist.gov/resources/n-lengths/list.html>`__,
    which in turn have been taken from Table 1 of Neutron News **3**, 26 (1992);
    `doi: 10.1080/10448639208218770 <https://doi.org/10.1080/10448639208218770>`_.

    Parameters
    ----------
    atom_types
        List of atomic species for which to retrieve scattering lengths.
    abundances
        Dict of the desired fractional abundance of each isotope for
        each species in the sample.  For example, to use an equal
        weighting of all isotopes of oxygen, one can write
        ``abundances['O'] = dict(16=1/3, 17=1/3, 18=1/3)``. Note that
        the abundance for any isotopes that are *not* included in this
        dict is automatically set to zero. In other words, you need to
        ensure that the abundances provided sum up to 1.  By default
        the neutron scattering lengths are weighted proportionally to
        the natural abundance of each isotope.
    """

    def __init__(
            self,
            atom_types: List[str],
            abundances: Dict[str, Dict[int, float]] = None,
    ):
        scat_lengths = _read_scattering_lengths()

        # Sub select only the relevant species
        scat_lengths = scat_lengths[scat_lengths.species.isin(atom_types)].reset_index()

        # Update the abundances if another weighting is desired
        if abundances is not None:
            for species in abundances:
                scat_lengths.loc[scat_lengths.species == species, 'abundance'] = 0
                for Z, frac in abundances[species].items():
                    match = (scat_lengths.species == species) & (scat_lengths.isotope == Z)
                    if not np.any(match):  # Check if any row+column matches
                        raise ValueError(f'No match in database for {species} and isotope {Z}')
                    scat_lengths.loc[match, 'abundance'] = frac

        self._scattering_lengths = scat_lengths

        # Check if any of the fetched scattering lengths is None,
        # indicating that it is missing in the experimental database.
        # Only raise an error if the desired abundance is greater than 0.
        nan_rows = scat_lengths[scat_lengths.isnull().any(axis=1) & (scat_lengths.abundance > 0.0)]
        if not nan_rows.empty:
            # Grab first offending entry
            row = nan_rows.iloc[0]
            raise ValueError(f'Non-zero abundance of {row.isotope}{row.species}'
                             ' with missing tabulated scattering length.')

        # Make sure abundances add up to 100%
        by_species = self._scattering_lengths.groupby('species')
        for species, species_df in by_species:
            if not np.isclose(species_df.abundance.sum(), 1):
                raise ValueError(f'Abundance values for {species} do not sum up to 1.0')

        # Compute scattering lengths weighted by abundance
        weights_coh = by_species.apply(
            lambda s: (s.b_coh * s.abundance)
            .sum().real,  # weights in dynasor can only be real atm
            include_groups=False
        ).to_dict()
        # First compute the average scattering length, then take the square
        # since the incoherent scattering lengths enter as b_incoh**2, but
        # dynasor only applies a single weighting factor w_incoh.
        weights_inc = by_species.apply(
            lambda s: ((s.b_inc * s.abundance).sum()**2).real,
            include_groups=False
        ).to_dict()

        supports_currents = False
        super().__init__(weights_coh, weights_inc, supports_currents)

    @property
    def abundances(self) -> Dict[str, Dict[int, float]]:
        abundance_dict = {}
        for (species), species_df in self._scattering_lengths.groupby('species'):
            abundance_dict[species] = {}
            for (isotope, abundance), _ in species_df.groupby(['isotope', 'abundance']):
                abundance_dict[species][isotope] = abundance
        return abundance_dict


def _read_scattering_lengths() -> DataFrame:
    """
    Extracts the scattering lengths from the file `neutron_scattering_lengths.json`
    for each of the supplied species. Scattering lengths are in units of fm.

    The scattering lengths have been extracted from the following NIST
    database: https://www.ncnr.nist.gov/resources/n-lengths/list.html,
    which in turn have been extracted from
    Neutron News **3**, No. 3***, 26 (1992).
    """
    data_file = files(__package__) / 'neutron_scattering_lengths.json'
    with open(data_file) as fp:
        scattering_lengths = json.load(fp)

    data = []
    for species in scattering_lengths:
        for isotope in scattering_lengths[species]:
            for fld in 'b_c b_i'.split():
                val = scattering_lengths[species][isotope][fld]
                if 'None' in val:
                    scattering_lengths[species][isotope][fld] = np.nan
                elif 'j' in val:
                    scattering_lengths[species][isotope][fld] = complex(val)
                else:
                    scattering_lengths[species][isotope][fld] = float(val)
            data.append(
                dict(
                    species=species,
                    isotope=int(isotope),
                    abundance=scattering_lengths[species][isotope]['abundance']
                    / 100,  # % -> fraction
                    b_coh=scattering_lengths[species][isotope]['b_c'],
                    b_inc=scattering_lengths[species][isotope]['b_i'],
                )
            )
    scattering_lengths = DataFrame.from_dict(data)
    return scattering_lengths
