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

        # will be set in _get_first
        self._id_to_internal = None  # numpy array mapping atom-id -> internal index (0..N-1)

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
        # Read first frame, update state of self, create indices etc
        step, N, cell, cols = self._read_frame_header()
        self._n_atoms = N
        self._step = step
        self._cols = cols
        self._cell = cell

        def _all_in_cols(keys):
            return all(k in cols for k in keys)

        # ---------------------------
        # 1) Figure out coordinate columns
        #    Allow missing x/y/z -> fill with 0
        # ---------------------------
        self._x_mode = None  # 'unwrapped' | 'wrapped' | 'scaled'
        self._coord_I = {'x': None, 'y': None, 'z': None}  # per-axis column index (or None)

        # Priority: xu/yu/zu > x/y/z > xs/ys/zs
        if _all_in_cols(('id', 'xu', 'yu', 'zu')) or ('id' in cols and ('xu' in cols or 'yu' in cols or 'zu' in cols)):
            self._x_mode = 'unwrapped'
            self._coord_I['x'] = cols.index('xu') if 'xu' in cols else None
            self._coord_I['y'] = cols.index('yu') if 'yu' in cols else None
            self._coord_I['z'] = cols.index('zu') if 'zu' in cols else None

        elif _all_in_cols(('id', 'x', 'y', 'z')) or ('id' in cols and ('x' in cols or 'y' in cols or 'z' in cols)):
            self._x_mode = 'wrapped'
            self._coord_I['x'] = cols.index('x') if 'x' in cols else None
            self._coord_I['y'] = cols.index('y') if 'y' in cols else None
            self._coord_I['z'] = cols.index('z') if 'z' in cols else None

        elif _all_in_cols(('id', 'xs', 'ys', 'zs')) or ('id' in cols and ('xs' in cols or 'ys' in cols or 'zs' in cols)):
            self._x_mode = 'scaled'
            self._coord_I['x'] = cols.index('xs') if 'xs' in cols else None
            self._coord_I['y'] = cols.index('ys') if 'ys' in cols else None
            self._coord_I['z'] = cols.index('zs') if 'zs' in cols else None

        else:
            # original error, but slightly clearer
            raise RuntimeError(
                "TRJ file must contain 'id' and at least one coordinate column among "
                "('x','y','z') or ('xu','yu','zu') or ('xs','ys','zs')."
            )

        if 'id' not in cols:
            raise RuntimeError("TRJ file must contain atom-id column named 'id'.")

        self._id_I = cols.index('id')

        # ---------------------------
        # 2) Velocities (optional)
        # ---------------------------
        if _all_in_cols(('vx', 'vy', 'vz')):
            self._v_I = array(deque(map(cols.index, ('vx', 'vy', 'vz'))))
        else:
            self._v_I = None

        # ---------------------------
        # 3) Type (optional)
        # ---------------------------
        self._type_I = cols.index('type') if 'type' in cols else None

        # ---------------------------
        # 4) Read frame data
        # ---------------------------
        data = array([list(map(float, self._fh.readline().split())) for _ in range(N)])
        # data.shape == (N, Ncols)

        ids = np.asarray(data[:, self._id_I], dtype=np.int_)
        # Build internal ordering: sort by atom-id (stable across frames)
        order = np.argsort(ids)
        # internal index per row of this frame:
        II = np.empty_like(ids)
        II[order] = np.arange(N, dtype=np.int_)

        # Create fast mapping id -> internal index (for next frames)
        max_id = int(ids.max())
        self._id_to_internal = np.full(max_id + 1, -1, dtype=np.int_)
        self._id_to_internal[ids[order]] = np.arange(N, dtype=np.int_)

        # Positions: fill missing axes with 0
        self._x = zeros((N, 3), dtype=float)

        def _axis_value(col_index, axis_scale=1.0):
            if col_index is None:
                return None
            return axis_scale * data[:, col_index]

        if self._x_mode == 'scaled':
            # scaled -> multiply each axis by cell diagonal (as original code did)
            diag = self._cell.diagonal()
            xvals = _axis_value(self._coord_I['x'], diag[0])
            yvals = _axis_value(self._coord_I['y'], diag[1])
            zvals = _axis_value(self._coord_I['z'], diag[2])
        else:
            xvals = _axis_value(self._coord_I['x'], 1.0)
            yvals = _axis_value(self._coord_I['y'], 1.0)
            zvals = _axis_value(self._coord_I['z'], 1.0)

        if xvals is not None:
            self._x[II, 0] = xvals
        # else: remain 0

        if yvals is not None:
            self._x[II, 1] = yvals

        if zvals is not None:
            self._x[II, 2] = zvals

        # Velocities
        if self._v_I is not None:
            self._v = zeros((N, 3), dtype=float)
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
        data = np.asarray(data, dtype=float)

        ids = np.asarray(data[:, self._id_I], dtype=np.int_)

        # Use id->internal mapping from first frame
        if self._id_to_internal is None:
            raise RuntimeError("Internal id mapping not initialized. _get_first must be called before _get_next.")

        max_id = self._id_to_internal.shape[0] - 1
        if ids.max() > max_id:
            # id range changed (unexpected). Expand mapping defensively.
            new_map = np.full(int(ids.max()) + 1, -1, dtype=np.int_)
            new_map[:self._id_to_internal.shape[0]] = self._id_to_internal
            self._id_to_internal = new_map

        II = self._id_to_internal[ids]
        if np.any(II < 0):
            raise RuntimeError("Found atom-id in later frame that did not exist in first frame (cannot map consistently).")

        # Fill positions (missing axes remain 0)
        # Start by zeroing only if you want strict behavior; here we keep previous values for missing axes?
        # For safety & determinism, we reset positions each frame to 0 then fill available axes.
        self._x[:, :] = 0.0

        if self._x_mode == 'scaled':
            diag = self._cell.diagonal()
            if self._coord_I['x'] is not None:
                self._x[II, 0] = data[:, self._coord_I['x']] * diag[0]
            if self._coord_I['y'] is not None:
                self._x[II, 1] = data[:, self._coord_I['y']] * diag[1]
            if self._coord_I['z'] is not None:
                self._x[II, 2] = data[:, self._coord_I['z']] * diag[2]
        else:
            if self._coord_I['x'] is not None:
                self._x[II, 0] = data[:, self._coord_I['x']]
            if self._coord_I['y'] is not None:
                self._x[II, 1] = data[:, self._coord_I['y']]
            if self._coord_I['z'] is not None:
                self._x[II, 2] = data[:, self._coord_I['z']]

        # Velocities
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
