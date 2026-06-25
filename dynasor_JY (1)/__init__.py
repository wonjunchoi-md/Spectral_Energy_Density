# -*- coding: utf-8 -*-

"""
dynasor module.
"""

from .correlation_functions import (compute_dynamic_structure_factors,
                                    compute_spectral_energy_density,
                                    compute_static_structure_factors)
from .qpoints import (get_spherical_qpoints,
                      get_supercell_qpoints_along_path)
from .sample import read_sample_from_npz
from .trajectory import Trajectory

__project__ = 'dynasor'
__description__ = 'Dynamical structure factors and correlation'
' functions from molecular dynamics trajectories'
__copyright__ = '2024'
__license__ = 'MIT'
__credits__ = ['The dynasor developers team']
__version__ = '2.1'
__maintainer__ = 'The dynasor developers team'
__status__ = 'Development Status :: 5 - Production/Stable'
__url__ = 'http://dynasor.materialsmodeling.org/'
__all__ = [
    'compute_dynamic_structure_factors',
    'compute_spectral_energy_density',
    'compute_static_structure_factors',
    'get_spherical_qpoints',
    'get_supercell_qpoints_along_path',
    'read_sample_from_npz',
    'Trajectory',
]
