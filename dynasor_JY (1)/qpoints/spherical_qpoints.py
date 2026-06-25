import itertools
import numpy as np
from numpy.typing import NDArray
from dynasor.logging_tools import logger


def get_spherical_qpoints(
        cell: NDArray[float],
        q_max: float,
        max_points: int = None,
        seed: int = 42,
) -> NDArray[float]:
    r"""Generates all q-points on the reciprocal lattice inside a given radius
    :attr:`q_max`.  This approach is suitable if an isotropic sampling of
    q-space is desired.  The function returns the resulting q-points in
    Cartesian coordinates as an ``Nx3`` array.

    If the number of generated q-points are large, points can be removed by
    specifying the :attr:`max_points`. The q-points will be randomly removed in
    such a way that the q-points inside are roughly uniformly distributed with
    respect to :math:`|q|`. If the number of q-points are binned w.r.t. their
    norm the function would increase quadratically up until some distance P
    from which point the distribution would be constant.

    Parameters
    ----------
    cell
        real cell with cell vectors as rows
    q_max
        maximum norm of generated q-points (in units of rad/Å, i.e. including factor of 2pi)
    max_points
        Optionally limit the set to __approximately__ :attr:`max_points` points
        by randomly removing points from a "fully populated mesh". The points
        are removed in such a way that for :math:`q > q_\mathrm{prune}`, the
        points will be radially uniformly distributed. The value of
        :math:`q_\mathrm{prune}` is calculated from :attr:`max_q`,
        :attr:`max_points`, and the shape of the cell.
    seed
        Seed used for stochastic pruning

    """

    # inv(A.T) == inv(A).T
    # The physicists reciprocal cell
    rec_cell = np.linalg.inv(cell.T) * 2 * np.pi

    # We want to find all points on the lattice defined by the reciprocal cell
    # such that all points within max_q are in this set
    inv_rec_cell = np.linalg.inv(rec_cell.T)  # cell / 2pi

    # h is the height of the rec_cell perpendicular to the other two vectors
    h = 1 / np.linalg.norm(inv_rec_cell, axis=1)

    # If a q_point has a coordinate larger than this number it must be further away than q_max
    N = np.ceil(q_max / h).astype(int)

    # Create all q-points within a sphere
    lattice_points = list(itertools.product(*[range(-n, n+1) for n in N]))
    q_points = lattice_points @ rec_cell

    # Calculate distances for pruning
    q_distances = np.linalg.norm(q_points, axis=1)  # Find distances

    # Sort distances and q-points based on distance
    argsort = np.argsort(q_distances)
    q_distances = q_distances[argsort]
    q_points = q_points[argsort]

    # Prune based on distances
    q_points = q_points[q_distances <= q_max]
    q_distances = q_distances[q_distances <= q_max]

    # Pruning based on max_points
    if max_points is not None and max_points < len(q_points):

        q_vol = np.linalg.det(rec_cell)

        q_prune = _get_prune_distance(max_points, q_max, q_vol)

        if q_prune < q_max:
            logger.info(f'Pruning q-points from the range {q_prune:.3} < |q| < {q_max}')

            # Keep point with probability min(1, (q_prune/|q|)^2) ->
            # aim for an equal number of points per equally thick "onion peel"
            # to get equal number of points per radial unit.
            p = np.ones(len(q_points))
            assert np.isclose(q_distances[0], 0)
            p[1:] = (q_prune / q_distances[1:]) ** 2

            rs = np.random.RandomState(seed)
            q_points = q_points[p > rs.rand(len(q_points))]

            logger.info(f'Pruned from {len(q_distances)} q-points to {len(q_points)}')

    return q_points


def _get_prune_distance(
        max_points: int,
        q_max: float,
        q_vol: float,
) -> NDArray[float]:
    r"""Determine distance in q-space beyond which to prune
    the q-point mesh to achieve near-isotropic sampling of q-space.

    If points are selected from the full mesh with probability
    :math:`\min(1, (q_\mathrm{prune} / |q|)^2)`, q-space will
    on average be sampled with an equal number of points per radial unit
    (for :math:`q > q_\mathrm{prune}`).

    The general idea is as follows.
    We know that the number of q-points inside a radius :math:`Q` is given by

    .. math:

        n = v^{-1} \int_0^Q dq 4 \pi q^2 = v^{-1} 4/3 \pi Q^3

    where :math:`v` is the volume of one q-point.  Now we want to find
    a distance :math:`P` such that if all points outside this radius
    are weighted by the function :math:`w(q)` the total number of
    q-points will equal the target :math:`N` (:attr:`max_points`)
    while the number of q-points increases linearly from :math:`P`
    outward. One additional constraint is that the weighting function
    must be 1 at :math:`P`. The weighting function which accomplishes
    this is :math:`w(q)=P^2/q^2`

    .. math:

        N = v^{-1} \left( \int_0^P 4 \pi q^2 + \int_P^Q 4 \pi q^2 P^2 / q^2 dq \right).

    This results in a `cubic equation <https://en.wikipedia.org/wiki/Cubic_equation>`_
    for :math:`P`, which is solved by this function.

    Parameters
    ----------
    max_points
        maximum number of resulting q-points; :math:`N` below
    max_q
        maximum q-value in the resulting q-point set; :math:`Q` below
    vol_q
        q-space volume for a single q-point
    """

    Q = q_max
    V = q_vol
    N = max_points

    # Coefs
    a = 1.0
    b = -3 / 2 * Q
    c = 0.0
    d = 3 / 2 * V * N / (4 * np.pi)

    # Eq tol solve
    def original_eq(x):
        return a * x**3 + b * x**2 + c * x + d
    # original_eq = lambda x:  a * x**3 + b * x**2 + c * x + d

    # Discriminant
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    D_t = - (4 * p**3 + 27 * q**2)
    if D_t < 0:
        return q_max

    x = Q * (np.cos(1 / 3 * np.arccos(1 - 4 * d / Q**3) - 2 * np.pi / 3) + 0.5)

    assert np.isclose(original_eq(x), 0), original_eq(x)

    return x
