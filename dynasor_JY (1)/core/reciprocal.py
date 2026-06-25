import numpy as np

from dynasor.core.rho_j_q_numba import rho_q as rho_q_numba
from dynasor.core.rho_j_q_numba import rho_j_q as rho_j_q_numba


def calc_rho_q(x, q):
    """Calculate rho(q) of particle coordinates x.

    Will call external function rho_q to calculate the
    particle density in q-space.
    Particle coordinates and q-space points of interest are
    passed as input via x and q, respectively.
    """

    assert x.shape[1] == 3
    assert q.shape[1] == 3

    Nx, _ = x.shape
    Nq, _ = q.shape

    rho_q = np.zeros(Nq, dtype=np.complex128)

    x = x.copy()  # Don't ask why
    rho_q_numba(x, q, rho_q)

    return rho_q


def calc_rho_j_q(x, v, q):
    """As calc_rho_q, but calculate also velocities in q-space
    """
    assert x.shape == v.shape

    assert x.shape[1] == 3
    assert v.shape[1] == 3
    assert q.shape[1] == 3

    Nx, _ = x.shape
    Nq, _ = q.shape

    rho_q = np.zeros(Nq, dtype=np.complex128)
    j_q = np.zeros((Nq, 3), dtype=np.complex128)

    x = x.copy()
    v = v.copy()

    rho_j_q_numba(x, v, q, rho_q, j_q)

    return rho_q, j_q
