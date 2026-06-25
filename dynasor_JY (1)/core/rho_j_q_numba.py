"""This module replaces the original c implementation of the reciprocal
densities and currents in dynasor with numba.

Numba is as of 2023 an ongoing project to create a JIT compiler frontend for
python code using the LLVM project as backend. Due to current limitations and
quirks of numba the code is not always straightforward. Typically the code
needs to be refactored in a trial and error process to get the expected
performance but should in the end be on the level of c.

Especially, numba makes very pessimistic assumptions about aliasing but this is
expected to change in the future. Also, in theory, via the llvm-lite interface
compilation flags should be passable to LLVM.
"""

import numpy as np
import numba


# This is often faster than calling np.dot for small arrays
# Calling this instead of manually inlining it actually incurs a small
# performance hit (<10%) with current numba (2023). It increases readability
# though and will probably sort itself out with later numba versions
@numba.njit(fastmath=True, nogil=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


# fastmath True makes the summation fast and also speeds up exponentiation
# nogil releases the python GIL, probably not neccesary here
@numba.njit(fastmath=True, nogil=True)
def rho_q_single(x: np.ndarray,
                 q: np.ndarray) -> complex:
    """Calculates the density at a single q-point

    Parameters
    ----------
    x
        positions as a (N, 3) array
    q
        single q point as a with shape (3,)

    Returns
    -------
    rho
        complex density at the specified q-point
    """
    Nx = len(x)

    assert x.shape == (Nx, 3)
    assert q.shape == (3,)

    rho = 0.0j
    for i in range(Nx):
        alpha = dot(x[i], q)
        rho += np.exp(1j * alpha)  # very expensive operation
    return rho


# parallel enables the numba.prange directive
@numba.njit(fastmath=True, nogil=True, parallel=True)
def rho_q(x: np.ndarray, q: np.ndarray, rho: np.ndarray):
    """Calculates the fourier transformed density

    The parallelization is over q-points. The density is calculated in-place.

    Parameters
    ----------
    x
        the positions as a float array with shape (``Nx``, 3)
    q
        the q points as a float array with shape (``Nq``, 3)
    rho
        density as a complex array of length ``Nq``
    """

    Nx = len(x)
    Nq = len(q)

    assert x.shape == (Nx, 3)
    assert q.shape == (Nq, 3)
    assert rho.shape == (Nq,)

    # Numba prange is like OMP
    for i in numba.prange(Nq):
        rho[i] = rho_q_single(x, q[i])


@numba.njit(fastmath=True, parallel=True, nogil=True)
def rho_j_q(x: np.ndarray, v: np.ndarray, q: np.ndarray,
            rho: np.ndarray, j_q: np.ndarray):
    """Calculates the fourier transformed density and current.

    The output is stored in the supplied output arrays ``rho`` and ``j_q``

    Parameters
    ----------
    x
        the positions as a float array with shape (``Nx``, 3)
    v
        the velocities as a float array with shape (``Nx``, 3)
    q
        the q points as a float array with shape (``Nq``, 3)
    rho
        density as a complex array of length ``Nq``
    j_q
        current as a complex array with shape (``Nq``, 3)
    """

    Nx = len(x)
    Nq = len(q)

    assert x.shape == (Nx, 3)
    assert v.shape == (Nx, 3)
    assert q.shape == (Nq, 3)
    assert rho.shape == (Nq,)
    assert j_q.shape == (Nq, 3)

    for qi in numba.prange(Nq):
        for xi in range(Nx):

            alpha = dot(x[xi], q[qi])
            exp_ialpha = np.exp(1.0j * alpha)

            rho[qi] += exp_ialpha

            for i in range(3):
                j_q[qi, i] += exp_ialpha * v[xi][i]
