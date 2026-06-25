from itertools import count
from dynasor.trajectory.abstract_trajectory_reader import AbstractTrajectoryReader
from dynasor.trajectory.trajectory_frame import ReaderFrame
from dynasor.logging_tools import logger
import MDAnalysis as mda
import warnings


class MDAnalysisTrajectoryReader(AbstractTrajectoryReader):
    """ Read a trajectory using the MDAnalysis Python library.

    Parameters
    ----------
    filename
        Name of input file.
    trajectory_format
        Type of trajectory. See MDAnalysis for the available formats.
    length_unit
        Unit of length for the input trajectory (``'Angstrom'``, ``'nm'``, ``'pm'``, ``'fm'``).
    time_unit
        Unit of time for the input trajectory (``'fs'``, ``'ps'``, ``'ns'``).
    """

    def __init__(self,
                 filename: str,
                 trajectory_format: str,
                 length_unit: str = 'Angstrom',
                 time_unit: str = 'fs'):

        self._open = True
        self._first_called = False

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='Guessed all Masses to 1.0')
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='Reader has no dt information, set to 1.0 ps')
            u = mda.Universe(filename, format=trajectory_format, convert_units=False)
        self._frame_index = count(0)
        self._trajectory = u.trajectory

        # Set atomic_indices dict, if possible
        try:
            self._atom_types = u.atoms.types
        except mda.exceptions.NoDataError:
            self._atom_types = None

        trajectory_length_unit = self._trajectory.units['length']
        trajectory_time_unit = self._trajectory.units['time']

        if trajectory_length_unit is None or trajectory_time_unit is None:  # No units from traj...
            if length_unit not in self.lengthunits_to_nm_table \
               or time_unit not in self.timeunits_to_fs_table:  # ... and incorrect units from user
                raise ValueError('Trajectory contains no unit information and specified units not '
                                 'recognized. Please check the available units.')

            else:                                        # ... but correct units from user
                def convert_units(ts):
                    length_scaling = mda.units.get_conversion_factor('length',
                                                                     length_unit,
                                                                     'Angstrom')
                    time_scaling = mda.units.get_conversion_factor('time',
                                                                   time_unit,
                                                                   'fs')
                    ts.positions *= length_scaling
                    ts.triclinic_dimensions *= length_scaling
                    if ts.has_velocities:
                        ts.velocities *= length_scaling/time_scaling
                    return ts
                self._trajectory.add_transformations(convert_units)
        else:                                                             # Units from trajectory
            if (length_unit != trajectory_length_unit and trajectory_length_unit is not None) or \
               (time_unit != trajectory_time_unit and trajectory_time_unit is not None):
                logger.warning(f'The units {length_unit} and {time_unit} were specified by user '
                               f'but the units {trajectory_length_unit} and {trajectory_time_unit} '
                               'were read from trajectory. Disregarding user-specified units and '
                               f'using {trajectory_length_unit} and {trajectory_time_unit}.')

            def convert_units(ts):
                length_scaling = mda.units.get_conversion_factor('length', trajectory_length_unit,
                                                                 'Angstrom')
                time_scaling = mda.units.get_conversion_factor('time', trajectory_time_unit, 'fs')
                ts.positions *= length_scaling
                ts.triclinic_dimensions *= length_scaling
                if ts.has_velocities:
                    ts.velocities *= length_scaling/time_scaling
                return ts
            self._trajectory.add_transformations(convert_units)

    def _get_next(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='Reader has no dt information, set to 1.0 ps')
            if self._first_called:
                self._trajectory.next()
            else:
                self._first_called = True
            self._positions = self._trajectory.ts.positions
            self._cell = self._trajectory.ts.triclinic_dimensions
            self._n_atoms = self._trajectory.ts.n_atoms
            if self._trajectory.ts.has_velocities:
                self._velocities = self._trajectory.ts.velocities
            else:
                self._velocities = None

    def __iter__(self):
        """ Iterates through the trajectory file, frame by frame. """
        return self

    def __next__(self):
        """ Gets next trajectory frame. """
        if not self._open:
            raise StopIteration

        self._get_next()

        if self._velocities is not None:
            frame = ReaderFrame(frame_index=next(self._frame_index),
                                cell=self._cell,
                                n_atoms=self._n_atoms,
                                positions=self._positions.copy(),
                                velocities=self._velocities.copy(),
                                atom_types=self._atom_types
                                )
        else:
            frame = ReaderFrame(frame_index=next(self._frame_index),
                                cell=self._cell,
                                n_atoms=self._n_atoms,
                                positions=self._positions.copy(),
                                atom_types=self._atom_types
                                )

        return frame

    def close(self):
        """ Closes down, release resources etc. """
        if self._open:
            self._trajectory.close()
            self._open = False
