import warnings
import numpy as np
from numpy.typing import NDArray


from dynasor.qpoints.tools import get_P_matrix, get_commensurate_lattice_points, find_on_line


class Lattice:

    def __init__(self, primitive: NDArray, supercell: NDArray):
        """Representation of a crystal supercell

        The supercell S is given by the primitive cell p and a repetition
        matrix P such that:

            dot(P, p) = S

        In this convention the cell vectors are row vectors of p and S as in
        ASE. An inverse cell is defined as:

            c_inv = inv(c).T

        and the reciprocal cell is defined as:

            c_rec = 2*pi*inv(c).T

        Notice that the inverse cell here is defined with the tranpose so that
        the lattic vectors of the inverse/reciprocal lattice are also row
        vectors. The above also implies:

            dot(P.T, S_inv) = p_inv

        The inverse supercell S_inv defines a new lattice in reciprocal space.
        Those inverse lattice points which resides inside the inverse primitive
        cell p_inv are called commensurate lattice points. These are typically
        the only points of interest in MD simulations from a crystallographic
        and lattice dynamics point of view.

        The convention taken here is that the reciprocal cell carries the 2pi
        factor onto the cartesian q-points. This is consistent with e.g.
        Kittel. The reduced coordinates are always with respect to the
        reciprocal primitive cell.

        Parameters
        ----------
        primitive
            cell metric of the primitive cell with lattice vectors as rows.
        supercell
            cell metric of the supercell with lattice vectors as rows
        """

        self._primitive = np.array(primitive)
        self._supercell = np.array(supercell)

        self._P = get_P_matrix(self.primitive, self.supercell)

        # As stated in the doc the P matrix relating the inverse cells are just P.T
        com = get_commensurate_lattice_points(self.P.T)
        self._qpoints = com @ self.reciprocal_supercell

    @property
    def primitive(self):
        """Returns the primitive cell with lattice vectors as rows"""
        return self._primitive

    @property
    def supercell(self):
        """Returns the supercell with lattice vectors as rows"""
        return self._supercell

    @property
    def reciprocal_primitive(self):
        """Returns inv(primitive).T so that the rows are the inverse lattice vectors"""
        return 2*np.pi * np.linalg.inv(self.primitive.T)  # inverse lattice as rows

    @property
    def reciprocal_supercell(self):
        """Returns inv(super).T so that the rows are the inverse lattice vectors"""
        return 2*np.pi * np.linalg.inv(self.supercell.T)  # reciprocal lattice as rows

    @property
    def P(self):
        """The P-matrix for this system, P @ primitive = supercell"""
        return self._P

    def __repr__(self):
        rep = (f'{self.__class__.__name__}('
               f'primitive={self.primitive.tolist()}, '
               f'supercell={self.supercell.tolist()})')
        return rep

    @property
    def qpoints(self):
        """Cartesian commensurate q-points"""
        return self._qpoints

    def __len__(self):
        return len(self._qpoints)

    def reduced_to_cartesian(self, qpoints):
        """Convert from reduced to cartesian coordinates"""
        return qpoints @ self.reciprocal_primitive

    def cartesian_to_reduced(self, qpoints):
        """Convert from Cartesian to reduced coordinates."""
        return np.linalg.solve(self.reciprocal_primitive.T, qpoints.T).T

    def make_path(self, start, stop):
        """Takes qpoints in reduced coordinates and returns all points in between

        Parameters
        ----------
        start
            coordinate of starting point in reduced inverse coordinates.
            e.g. a zone mode is given as (0.5, 0, 0), (0,0,0) == (1,0,0) etc.
        stop
            stop position

        Returns
        -------
        qpoints
            coordinates of commensurate points along path in cartesian reciprocals
        dists
            fractional distance along path
        """
        start = np.array(start)
        stop = np.array(stop)
        fracs = find_on_line(start, stop, self.P.T)
        if not len(fracs):
            warnings.warn('No q-points along path!')
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0,))
        points = np.array([start + float(f) * (stop - start) for f in fracs])
        qpoints = self.reduced_to_cartesian(points)
        dists = np.array([float(f) for f in fracs])
        return qpoints, dists
