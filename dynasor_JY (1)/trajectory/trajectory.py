__all__ = ['Trajectory', 'WindowIterator']

import numpy as np

from collections import deque
from itertools import islice, chain
from os.path import isfile
from typing import Callable, Dict, Union, List

from dynasor.trajectory.atomic_indices import parse_gromacs_index_file
from dynasor.trajectory.ase_trajectory_reader import ASETrajectoryReader
from dynasor.trajectory.extxyz_trajectory_reader import ExtxyzTrajectoryReader
from dynasor.trajectory.lammps_trajectory_reader import LammpsTrajectoryReader
from dynasor.trajectory.mdanalysis_trajectory_reader import MDAnalysisTrajectoryReader
from dynasor.trajectory.trajectory_frame import TrajectoryFrame
from dynasor.logging_tools import logger


class Trajectory:
    """Instances of this class hold trajectories in a format suitable for
    the computation of correlation functions.  They behave as
    iterators, where each step returns the next frame as a
    :class:`TrajectoryFrame` object.  The latter hold information
    regarding atomic positions, types, and velocities.

    Parameters
    ----------
    filename
        Name of input file
    trajectory_format
        Type of trajectory. Possible values are:
        ``'lammps_internal'``, ``'extxyz'``, ``'ase'`` or one of the formats supported by
        `MDAnalysis <https://www.mdanalysis.org/>`_ (except for ``'lammpsdump'``,
        which can be called by specifying ``'lammps_mdanalysis'`` to avoid ambiguity)
    atomic_indices
        Specify which indices belong to which atom type. Can be
        (1) a dictionary where the keys specicy species and the values are list of atomic indices,
        (2) ``'read_from_trajectory'``, in which case the species are read from the trajectory or
        (3) the path to an index file.
    length_unit
        Length unit of trajectory (``'Angstrom'``, ``'nm'``, ``'pm'``, ``'fm'``). Necessary for
        correct conversion to internal dynasor units if the trajectory file does not contain unit
        information.
    time_unit
        Time unit of trajectory (``'fs'``, ``'ps'``, ``'ns'``). Necessary for correct conversion to
        internal dynasor units if the trajectory file does not contain unit information.
    frame_start
        First frame to read; must be larger or equal ``0``.
    frame_stop
        Last frame to read. By default (``None``) the entire trajectory is read.
    frame_step
        Read every :attr:`frame_step`-th step of the input trajectory.
        By default (``1``) every frame is read. Must be larger than ``0``.

    """
    def __init__(
            self,
            filename: str,
            trajectory_format: str,
            atomic_indices: Union[str, Dict[str, List[int]]] = None,
            length_unit: str = 'Angstrom',
            time_unit: str = 'fs',
            frame_start: int = 0,
            frame_stop: int = None,
            frame_step: int = 1
    ):

        if frame_start < 0:
            raise ValueError('frame_start should be positive')
        if frame_step < 0:
            raise ValueError('frame_step should be positive')

        self._frame_start = frame_start
        self._frame_step = frame_step
        self._frame_stop = frame_stop

        # setup trajectory reader
        if not isfile(filename):
            raise IOError(f'File {filename} does not exist')
        self._filename = filename

        if trajectory_format == 'lammps_internal':
            reader = LammpsTrajectoryReader
        elif trajectory_format == 'extxyz':
            reader = ExtxyzTrajectoryReader
        elif trajectory_format == 'lammps_mdanalysis':
            reader = MDAnalysisTrajectoryReader
            trajectory_format = 'lammpsdump'
        elif trajectory_format == 'ase':
            reader = ASETrajectoryReader
        elif trajectory_format == 'lammps':
            raise IOError('Ambiguous trajectory format, '
                          'did you mean lammps_internal or lammps_mdanalysis?')
        else:
            reader = MDAnalysisTrajectoryReader

        logger.debug(f'Using trajectory reader: {reader.__name__}')
        if reader == MDAnalysisTrajectoryReader:
            self._reader_obj = reader(self._filename, trajectory_format,
                                      length_unit=length_unit, time_unit=time_unit)
        else:
            self._reader_obj = reader(self._filename, length_unit=length_unit, time_unit=time_unit)

        # Get two frames to set cell etc.
        frame0 = next(self._reader_obj)
        frame1 = next(self._reader_obj)
        self._cell = frame0.cell
        self._n_atoms = frame0.n_atoms

        # Make sure cell is not changed during consecutive frames
        if not np.allclose(frame0.cell, frame1.cell):
            raise ValueError('The cell changes between the first and second frame. '
                             'The concept of q-points becomes muddy if the simulation cell is '
                             'changing, such as during NPT MD simulations, so trajectories where '
                             'the cell changes are not supported by dynasor.')

        # setup iterator slice (reuse frame0 and frame1 via chain)
        self.number_of_frames_read = 0
        self.current_frame_index = 0
        self._reader_iter = islice(chain([frame0, frame1], self._reader_obj),
                                   self._frame_start, self._frame_stop, self._frame_step)

        # setup atomic indices
        if atomic_indices is None:  # Default behaviour
            atomic_indices = {'X': np.arange(0, self.n_atoms)}
        elif isinstance(atomic_indices, str):  # Str input
            if atomic_indices == 'read_from_trajectory':
                if frame0.atom_types is None:
                    raise ValueError('Could not read atomic indices from the trajectory.')
                else:
                    uniques = np.unique(frame0.atom_types)
                    atomic_indices = {uniques[i]: (frame0.atom_types == uniques[i]).nonzero()[0]
                                      for i in range(len(uniques))}
            else:
                atomic_indices = parse_gromacs_index_file(atomic_indices)
        elif isinstance(atomic_indices, dict):  # Dict input
            pass
        else:
            raise ValueError('Could not understand atomic_indices.')
        self._atomic_indices = atomic_indices

        # sanity checks for atomic_indices
        for key, indices in self._atomic_indices.items():
            if np.max(indices) > self.n_atoms:
                raise ValueError('maximum index in atomic_indices exceeds number of atoms')
            if np.min(indices) < 0:
                raise ValueError('minimum index in atomic_indices is negative')
            if '_' in key:
                # Since '_' is what we use to distinguish atom types in the results, e.g. Sqw_Cs_Pb
                raise ValueError('The char "_" is not allowed in atomic_indices.')

        # log info on trajectory and atom types etc
        logger.info(f'Trajectory file: {self.filename}')
        logger.info(f'Total number of particles: {self.n_atoms}')
        logger.info(f'Number of atom types: {len(self.atom_types)}')
        for atom_type, indices in self._atomic_indices.items():
            logger.info(f'Number of atoms of type {atom_type}: {len(indices)}')
        logger.info(f'Simulation cell (in Angstrom):\n{str(self._cell)}')

    def __iter__(self):
        return self

    def __next__(self):
        frame = next(self._reader_iter)
        new_frame = TrajectoryFrame(self.atomic_indices, frame.frame_index, frame.positions,
                                    frame.velocities)
        self.number_of_frames_read += 1
        self.current_frame_index = frame.frame_index
        return new_frame

    def __str__(self) -> str:
        s = ['Trajectory']
        s += ['{:12} : {}'.format('filename', self.filename)]
        s += ['{:12} : {}'.format('natoms', self.n_atoms)]
        s += ['{:12} : {}'.format('frame_start', self._frame_start)]
        s += ['{:12} : {}'.format('frame_stop', self._frame_stop)]
        s += ['{:12} : {}'.format('frame_step', self.frame_step)]
        s += ['{:12} : {}'.format('frame_index', self.current_frame_index)]
        s += ['{:12} : [{}\n                {}\n                {}]'
              .format('cell', self.cell[0], self.cell[1], self.cell[2])]
        return '\n'.join(s)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        s = [f'<h3>{self.__class__.__name__}</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Value</th></tr></thead>']
        s += ['<tbody>']
        s += [f'<tr"><td style="text-align: left;">File name</td><td>{self.filename}</td></tr>']
        s += [f'<tr><td style="text-align: left;">Number of atoms</td><td>{self.n_atoms}</td></tr>']
        s += [f'<tr><td style="text-align: left;">Cell metric</td><td>{self.cell}</td></tr>']
        s += [f'<tr><td style="text-align: left;">Frame step</td><td>{self.frame_step}</td></tr>']
        s += [f'<tr><td style="text-align: left;">Atom types</td><td>{self.atom_types}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return '\n'.join(s)

    @property
    def cell(self):
        """ Simulation cell """
        return self._cell

    @property
    def n_atoms(self):
        """ Number of atoms """
        return self._n_atoms

    @property
    def filename(self):
        """ The trajectory filename """
        return self._filename

    @property
    def atomic_indices(self):
        """ Return copy of index arrays """
        atomic_indices = dict()
        for name, inds in self._atomic_indices.items():
            atomic_indices[name] = inds.copy()
        return atomic_indices

    @property
    def atom_types(self) -> List[str]:
        return sorted(self._atomic_indices.keys())

    @property
    def frame_step(self):
        """ Frame to access, trajectory will return every :attr:`frame_step`-th snapshot """
        return self._frame_step


def consume(iterator, n):
    """ Advance the iterator by :attr:`n` steps. If :attr:`n` is ``None``, consume entirely. """
    # From the python.org
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


class WindowIterator:
    """Sliding window iterator.

    Returns consecutive windows (a window is represented as a list
    of objects), created from an input iterator.

    Parameters
    ----------
    itraj
        Trajectory object
    width
        Length of window (``window_size`` + 1)
    window_step
        Distance between the start of two consecutive window frames
    element_processor
        Enables processing each non-discarded object; useful if ``window_step >
        width`` and ``map_item`` is expensive (as compared to directly passing
        ``map(fun, itraj)`` as ``itraj``); if ``window_step < width``, you could as
        well directly pass ``map(fun, itraj)``.
    """
    def __init__(self,
                 itraj: Trajectory,
                 width: int,
                 window_step: int = 1,
                 element_processor: Callable = None):

        self._raw_it = itraj
        if element_processor:
            self._it = map(element_processor, self._raw_it)
        else:
            self._it = self._raw_it
        assert window_step >= 1
        assert width >= 1
        self.width = width
        self.window_step = window_step
        self._window = None

    def __iter__(self):
        return self

    def __next__(self):
        """ Returns next element in sequence. """
        if self._window is None:
            self._window = deque(islice(self._it, self.width), self.width)
        else:
            if self.window_step >= self.width:
                self._window.clear()
                consume(self._raw_it, self.window_step - self.width)
            else:
                for _ in range(min((self.window_step, len(self._window)))):
                    self._window.popleft()
            for f in islice(self._it, min((self.window_step, self.width))):
                self._window.append(f)

        if len(self._window) == 0:
            raise StopIteration

        return list(self._window)
