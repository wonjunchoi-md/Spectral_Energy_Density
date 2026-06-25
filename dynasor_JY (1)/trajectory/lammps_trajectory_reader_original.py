import numpy as np
import re

from collections import deque
from dynasor.trajectory.abstract_trajectory_reader import AbstractTrajectoryReader
from dynasor.trajectory.trajectory_frame import ReaderFrame
from itertools import count
from numpy import array, arange, zeros


class LammpsTrajectoryReader(AbstractTrajectoryReader):
    """Read LAMMPS trajectory file

    This is a naive (and comparatively slow) implementation,
    written entirely in python.

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
        time_unit: str = 'fs'
    ):

        if filename.endswith('.gz'):
            import gzip
            self._fh = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            import bz2
            self._fh = bz2.open(filename, 'rt')
        else:
            self._fh = open(filename, 'r')

        self._open = True
        regexp = r'^ITEM: (TIMESTEP|NUMBER OF ATOMS|BOX BOUNDS|ATOMS) ?(.*)$'
        self._item_re = re.compile(regexp)

        self._first_called = False
        self._frame_index = count(0)

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

    # ITEM: TIMESTEP
    # 81000
    # ITEM: NUMBER OF ATOMS
    # 1536
    # ITEM: BOX BOUNDS pp pp pp
    # 1.54223 26.5378
    # 1.54223 26.5378
    # 1.54223 26.5378
    # ITEM: ATOMS id type x y z vx vy vz
    # 247 1 3.69544 2.56202 3.27701 0.00433856 -0.00099307 -0.00486166
    # 249 2 3.73324 3.05962 4.14359 0.00346029 0.00332502 -0.00731005
    # 463 1 3.5465 4.12841 5.34888 0.000523332 0.00145597 -0.00418675

    def _read_frame_header(self):
        while True:
            L = self._fh.readline()
            m = self._item_re.match(L)
            if not m:
                if L == '':
                    self._fh.close()
                    self._open = False
                    raise StopIteration
                if L.strip() == '':
                    continue
                raise IOError('TRJ_reader: Failed to parse TRJ frame header')
            if m.group(1) == 'TIMESTEP':
                step = int(self._fh.readline())
            elif m.group(1) == 'NUMBER OF ATOMS':
                n_atoms = int(self._fh.readline())
            elif m.group(1) == 'BOX BOUNDS':
                bbounds = [deque(map(float, self._fh.readline().split()))
                           for _ in range(3)]
                x = array(bbounds)
                cell = np.diag(x[:, 1] - x[:, 0])
                if x.shape == (3, 3):
                    cell[1, 0] = x[0, 2]
                    cell[2, 0] = x[1, 2]
                    cell[2, 1] = x[2, 2]
                elif x.shape != (3, 2):
                    raise IOError('TRJ_reader: Malformed cell bounds')
            elif m.group(1) == 'ATOMS':
                cols = tuple(m.group(2).split())
                # At this point, there should be only atomic data left
                return (step, n_atoms, cell, cols)

    def _get_first(self):
        # Read first frame, update state of self, create indeces etc
        step, N, cell, cols = self._read_frame_header()
        self._n_atoms = N
        self._step = step
        self._cols = cols
        self._cell = cell

        def _all_in_cols(keys):
            for k in keys:
                if k not in cols:
                    return False
            return True

        self._x_map = None
        if _all_in_cols(('id', 'xu', 'yu', 'zu')):
            self._x_I = array(deque(map(cols.index, ('xu', 'yu', 'zu'))))
        elif _all_in_cols(('id', 'x', 'y', 'z')):
            self._x_I = array(deque(map(cols.index, ('x', 'y', 'z'))))
        elif _all_in_cols(('id', 'xs', 'ys', 'zs')):
            self._x_I = array(deque(map(cols.index, ('xs', 'ys', 'zs'))))
            _x_factor = self._cell.diagonal()
            # xs.shape == (n, 3)
            self._x_map = lambda xs: xs * _x_factor
        else:
            raise RuntimeError('TRJ file must contain at least atom-id, x, y,'
                               ' and z coordinates to be useful.')
        self._id_I = cols.index('id')

        if _all_in_cols(('vx', 'vy', 'vz')):
            self._v_I = array(deque(map(cols.index, ('vx', 'vy', 'vz'))))
        else:
            self._v_I = None

        if 'type' in cols:
            self._type_I = cols.index('type')
        else:
            self._type_I = None

        data = array([list(map(float, self._fh.readline().split())) for _ in range(N)])
        # data.shape == (N, Ncols)
        II = np.asarray(data[:, self._id_I], dtype=np.int_)
        # Unless dump is done for group 'all' ...
        II[np.argsort(II)] = arange(len(II))
        self._x = zeros((N, 3))
        if self._x_map is None:
            self._x[II] = data[:, self._x_I]
        else:
            self._x[II] = self._x_map(data[:, self._x_I])
        if self._v_I is not None:
            self._v = zeros((N, 3))
            self._v[II] = data[:, self._v_I]

        self._first_called = True

    def _get_next(self):
        # get next frame, update state of self
        step, N, cell, cols = self._read_frame_header()
        assert self._n_atoms == N
        assert self._cols == cols
        self._step = step
        self._cell = cell

        data = array([deque(map(float, self._fh.readline().split()))
                      for _ in range(N)])
        II = np.asarray(data[:, self._id_I], dtype=np.int_) - 1
        if self._x_map is None:
            self._x[II] = data[:, self._x_I]
        else:
            self._x[II] = self._x_map(data[:, self._x_I])
        if self._v_I is not None:
            self._v[II] = data[:, self._v_I]

    def __iter__(self):
        return self

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __next__(self):
        if not self._open:
            raise StopIteration

        if self._first_called:
            self._get_next()
        else:
            self._get_first()

        if self._v_I is not None:
            frame = ReaderFrame(frame_index=next(self._frame_index),
                                n_atoms=int(self._n_atoms),
                                cell=self.x_factor * self._cell.copy('F'),
                                positions=self.x_factor * self._x,
                                velocities=self.v_factor * self._v
                                )
        else:
            frame = ReaderFrame(frame_index=next(self._frame_index),
                                n_atoms=int(self._n_atoms),
                                cell=self.x_factor * self._cell.copy('F'),
                                positions=self.x_factor * self._x
                                )

        return frame
