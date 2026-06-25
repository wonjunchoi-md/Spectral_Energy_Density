"""
Module containing convenient unit conversion factors, including

* ``radians_per_fs_to_THz``
* ``radians_per_fs_to_meV``
* ``radians_per_fs_to_invcm``
* ``THz_to_meV``
* ``THz_to_invcm``
* ``meV_to_invcm``
* ``Dalton_to_dmu``  dmu is Dynasor Mass Unit


Here, for example, ``radians_per_fs_to_invcm`` can be used to convert an angular frequency in units
of radians/fs to a frequency in 1/cm, as demonstrated by the code snippet below.

.. highlight:: python
.. code-block:: python

    # converting the angular frequencies (omega) in a Sample object to frequencies in inverse cm
    from dynasor.units import radians_per_fs_to_invcm
    frequencies_invcm = sample.omega * radians_per_fs_to_invcm

    # converting frequencies from inverse cm to meV
    from dynasor.units import meV_to_invcm
    frequencies_meV = frequencies_invcm / meV_to_invcm

"""
from math import pi
from ase.units import _c, invcm, fs

# Frequencies
meV_to_invcm = 1 / invcm / 1e3
THz_to_invcm = 1e12 / _c / 1e2
THz_to_meV = 1e13 * invcm / _c

# Angular frequencies
radians_per_fs_to_THz = 1000 / 2 / pi
radians_per_fs_to_meV = radians_per_fs_to_THz * THz_to_meV
radians_per_fs_to_invcm = radians_per_fs_to_THz * THz_to_invcm

# Mass
Dalton_to_dmu = 1/fs**2
