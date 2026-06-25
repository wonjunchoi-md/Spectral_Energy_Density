import numpy as np
from ase import Atoms


def align_structure(atoms: Atoms, atol: float = 1e-5):
    """

    Rotate and realign atoms object such that
    * the first cell vector points along the x-directon
    * the second cell vector lies in the xy-plane

    Modifies input ``atoms`` object in place.

    Parameters
    ----------
    atoms
        input structure to be rotated aligned wtih the x,y,z coordinte system
    atol
        absolute tolerance used for sanity checking the cell
    """
    _align_a_onto_xy(atoms, atol)
    _align_a_onto_x(atoms, atol)
    _align_b_onto_xy(atoms, atol)


def _align_a_onto_xy(atoms, atol):
    """Rotate cell so that a is in the xy-plane"""

    # get angle towards xy
    # will break if a is along z
    assert np.any(atoms.cell[0, :2])

    cell = atoms.cell.array.copy()

    a = cell[0]
    a_xy = a.copy()
    a_xy[2] = 0  # projection of a onto xy-plane

    # angle between a and xy-plane
    cosa = np.dot(a, a_xy) / np.linalg.norm(a) / np.linalg.norm(a_xy)

    # cosa should be in the interval (0, 1]
    assert not np.isclose(cosa, 0)
    if cosa > 1:
        assert np.isclose(cosa, 1)
    cosa = min(cosa, 1)
    cosa = max(cosa, 0)

    # angle between a and xy-plane in degs
    angle_xy_deg = np.rad2deg(np.arccos(cosa))

    # get unit vector to rotate around
    vec = np.cross(a_xy, [0, 0, 1])
    vec = vec / np.linalg.norm(vec)
    assert vec[2] == 0

    # Determine if the rotation should be positive or negative depending on
    # whether a is pointing in the +z or -z direction
    sign = -1 if a[2] > 0 else +1

    # rotate
    atoms.rotate(sign * angle_xy_deg, vec, rotate_cell=True)

    assert np.isclose(atoms.cell[0, 2], 0, atol=atol, rtol=0), atoms.cell


def _align_a_onto_x(atoms, atol):
    assert np.isclose(atoms.cell[0, 2], 0, atol=atol, rtol=0)  # make sure a is in xy-plane

    a = atoms.cell[0]
    a_x = a[0]
    a_y = a[1]

    # angle between a and x-axis (a is already in xy-plane)

    # tan = y / x -> angle = arctan y / x "=" atan2(y, x)
    angle_rad = np.arctan2(a_y, a_x)
    angle_deg = np.rad2deg(angle_rad)

    atoms.rotate(-angle_deg, [0, 0, 1], rotate_cell=True)

    assert np.isclose(atoms.cell[0, 1], 0, atol=atol, rtol=0), atoms.cell
    assert np.isclose(atoms.cell[0, 2], 0, atol=atol, rtol=0), atoms.cell


def _align_b_onto_xy(atoms, atol):
    assert np.isclose(atoms.cell[0, 1], 0, atol=atol, rtol=0)  # make sure a is along x
    assert np.isclose(atoms.cell[0, 2], 0, atol=atol, rtol=0)  # make sure a is along x

    # rotate so that b is in xy plane
    # project b onto the yz-plane
    b = atoms.cell[1]
    b_y = b[1]
    b_z = b[2]
    angle_rad = np.arctan2(b_z, b_y)
    angle_deg = np.rad2deg(angle_rad)

    atoms.rotate(-angle_deg, [1, 0, 0], rotate_cell=True)

    assert np.isclose(atoms.cell[0, 1], 0, atol=atol, rtol=0)  # make sure a is in xy-plane
    assert np.isclose(atoms.cell[0, 2], 0, atol=atol, rtol=0)  # make sure a is in xy-plane
    assert np.isclose(atoms.cell[1, 2], 0, atol=atol, rtol=0), atoms.cell
