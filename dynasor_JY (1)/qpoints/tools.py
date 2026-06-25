import itertools
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from ase import Atoms


def get_supercell_qpoints_along_path(
        path: List[Tuple[str, str]],
        coordinates: Dict[str, NDArray[float]],
        primitive_cell: NDArray[float],
        super_cell: NDArray[float]) -> List[NDArray[float]]:
    r"""
    Returns the q-points commensurate with the given supercell along the specific path.

    Parameters
    ----------
    path
        list of pairs of q-point labels
    coordinates
        dict with q-point labels and coordinates as keys and values, respectively;
        there must be one entry for each q-point label used in :attr:`path`
    primitive_cell
        cell metric of the primitive cell with lattice vectors as rows
    super_cell
        cell metric of the supercell with lattice vectors as rows

    Returns
    -------
    supercell_paths
        A list of the accessible q-point coordinates along the specified segment

    Example
    --------
    The following example illustrates how to retrieve the q-points that
    can be sampled using a supercell comprising :math:`6 \times 6 \times 6`
    conventional (4-atom) unit cells of FCC Al along the path X-:math:`\Gamma`-L.

    >>> import numpy as np
    >>> from ase.build import bulk
    >>> from dynasor.qpoints import get_supercell_qpoints_along_path
    >>> prim = bulk('Al', 'fcc', a=4.0)
    >>> supercell = bulk('Al', 'fcc', a=4.0, cubic=True).repeat(6)
    >>> path = [('X', 'G'), ('G', 'L'), ('L', 'W')]
    >>> coordinates = dict(X=[0.5, 0.5, 0], G=[0, 0, 0],
    ...                    L=[0.5, 0.5, 0.5], W=[0.5, 0.25, 0.75])
    >>> qpoints = get_supercell_qpoints_along_path(path, coordinates, prim.cell, supercell.cell)

    """
    from .lattice import Lattice
    lat = Lattice(primitive_cell, super_cell)

    for lbl in np.array(path).flatten():
        if lbl not in coordinates:
            raise ValueError(f'Unknown point in path: {lbl}')

    # build the segments
    supercell_paths = []
    for k, (l1, l2) in enumerate(path):
        q1 = np.array(coordinates[l1], dtype=float)
        q2 = np.array(coordinates[l2], dtype=float)
        dynasor_path, _ = lat.make_path(q1, q2)
        supercell_paths.append(dynasor_path)
    return supercell_paths


def find_on_line(start: NDArray, stop: NDArray, P: NDArray):
    """Find fractional distances between start and stop combatible with P

    A supercell is defined by P @ c = S for some repetition matrix P and we
    want to find fractions so that

        [start + f * (stop - start)] @ P = n

    Parameters
    ----------
    start
        start of line in reduced supercell coordinates
    stop
        end of line in reduced supercell coordinates
    P
        repetion matrix defining the supercell
    """

    if np.allclose(start, stop):
        return [Fraction(0, 1)]

    start = np.array([Fraction(s).limit_denominator() for s in start])
    stop = np.array([Fraction(s).limit_denominator() for s in stop])

    A = start @ P
    B = (stop - start) @ P

    fracs = None
    for a, b in zip(A, B):
        fs = solve_Diophantine(a, b)
        if fs is None:  # "inf" solutions
            continue
        elif fs == []:  # No solutions
            return []
        fracs = set(fs) if fracs is None else fracs.intersection(fs)
    return sorted(fracs)


def solve_Diophantine(a: Fraction, b: Fraction) -> List[Fraction]:
    """Solve n = a + xb for all n in Z and a,b in Q such that 0 <= x <= 1"""

    if b == 0:
        if a.denominator == 1:
            return None
        else:
            return []

    if b < 0:
        right = np.ceil(a)
        left = np.floor(a + b)
    else:
        left = np.floor(a)
        right = np.ceil(a + b)

    ns = np.arange(left, right + 1)
    fracs = [Fraction(n - a, b) for n in ns]
    fracs = [f for f in fracs if 0 <= f <= 1]

    return fracs


def det(A):
    """Determinant of an integer matrix using Laplace cofactor expansion"""
    if len(A) == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    d = 0
    for i, B in enumerate(A[0]):  # along first row
        minor = np.hstack([A[1:, :i], A[1:, i+1:]])
        d += (-1)**i * B * det(minor)
    assert np.isclose(d, np.linalg.det(A))
    return d


def inv(A):
    """Takes the inverse of an integer 3x3 matrix based on Cayley-Hamilton"""

    detx2 = det(A) * 2  # Denominator "determinant times two"

    # Numerator
    numerator = ((np.trace(A)**2 - np.trace(A @ A)) * np.diag([1, 1, 1])
                 - 2 * A * np.trace(A)
                 + 2 * A @ A)

    # We want the sign to be in the Numerator
    if detx2 < 0:
        detx2 = -detx2
        numerator = -numerator

    inverse = numerator / detx2
    assert np.allclose(inverse, np.linalg.inv(A))

    # Return inverse, numerator (int matrix) and denominator (int)
    return inverse, numerator, detx2


def get_P_matrix(c, S):
    """ P c = S  ->  c.T P.T = S.T

    The P matrix must be an integer matrix
    """
    PT = np.linalg.solve(c.T, S.T)
    P_float = PT.T
    P = np.round(P_float).astype(int)
    if not np.allclose(P_float, P) or not np.allclose(P @ c, S):
        raise ValueError(
            f'Please check that the supercell metric ({S}) is related to the'
            f' the primitive cell {c} by an integer transformation matrix.')
    return P


def get_commensurate_lattice_points(P: NDArray) -> NDArray:
    """Return commensurate points for a supercell defined by repetition matrix P

    Finds all n such that n = f P where f is between 0 and 1

    Parameters
    ----------
    P
        the repetion matrix relating the primitive and supercell

    Returns
    -------
    lattice_points
        the commensurate lattice points
    """

    n_max = np.where(P > 0, P, 0).sum(axis=0) + 1
    n_min = np.where(P < 0, P, 0).sum(axis=0)

    ranges = [np.arange(*n) for n in zip(n_min, n_max)]

    inv_P_matrix, num, den = inv(P)

    lattice_points = []
    for lp in itertools.product(*ranges):
        s = lp @ num  # here we skip the denominator to keep everything integer
        # the denominator is also integer so no numerics here
        if np.all(s >= 0) and np.all(s < den):
            lattice_points.append(lp)

    lattice_points = np.array(lattice_points)

    # Begin sane checks...

    # No duplicates
    assert len(lattice_points) == len(np.unique(lattice_points, axis=0))

    # Did we get everyone?
    assert len(lattice_points) == abs(det(P))

    return lattice_points


def get_index_offset(supercell: Atoms, prim: Atoms, atol=1e-3, rtol=0.0):
    """
    Get the basis index and primitive cell offsets for a supercell
    """

    if len(prim) > len(supercell):
        raise ValueError('prim contains more atoms than supercell')

    index, offset = [], []
    for pos in supercell.positions:
        spos = np.linalg.solve(prim.cell.T, pos)
        for i, spos2 in enumerate(prim.get_scaled_positions()):
            off = spos - spos2
            off_round = np.round(off)
            if not np.allclose(off, off_round, atol=atol, rtol=rtol):
                continue
            index.append(i)
            off = off_round.astype(int)
            assert np.allclose(off, off_round)
            offset.append(off)
            break
        else:
            raise ValueError('prim not compatible with supercell')

    index, offset = np.array(index), np.array(offset)
    return index, offset
