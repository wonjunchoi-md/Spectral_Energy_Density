import numpy as np
from ase import io
from dynasor.trajectory.abstract_trajectory_reader import AbstractTrajectoryReader
from dynasor.trajectory.trajectory_frame import ReaderFrame
from itertools import count


class ASETrajectoryReader(AbstractTrajectoryReader):
    """Read ASE trajectory file

    ...

    Parameters
    ----------
    filename
        Name of input file.
    length_unit
        Unit of length for the input trajectory (``'Angstrom'``, ``'nm'``, ``'pm'``, ``'fm'``).
    time_unit
        Unit of time for the input trajectory (``'fs'``, ``'ps'``, ``'ns'``).
    """

    def __init__(
        self,
        filename: str,
        length_unit: str = 'Angstrom',
        time_unit: str = 'fs',
    ):
        self._frame_index = count(0)
        self._atoms = io.iread(filename, index=':')

        # setup units
        if length_unit not in self.lengthunits_to_nm_table:
            raise ValueError(f'Specified length unit {length_unit} is not an available option.')
        else:
            self.x_factor = self.lengthunits_to_nm_table[length_unit]
        if time_unit not in self.timeunits_to_fs_table:
            raise ValueError(f'Specified time unit {time_unit} is not an available option.')
        else:
            self.t_factor = self.timeunits_to_fs_table[time_unit]
        self.v_factor = self.x_factor / self.t_factor

    def __iter__(self):
        return self

    def close(self):
        pass

    def __next__(self):
        ind = next(self._frame_index)
        a = next(self._atoms)
        if 'momenta' in a.arrays:
            vel = self.v_factor * a.get_velocities()
        else:
            vel = None
        return ReaderFrame(
            frame_index=ind,
            n_atoms=len(a),
            cell=self.x_factor * a.cell.array.copy('F'),
            positions=self.x_factor * a.get_positions(),
            velocities=vel,
            atom_types=np.array(list(a.symbols)),
        )
