"""
This module provides an implementation of Filon's integration formula.
For information about Filon's formula, see e.g.
`Abramowitz and Stegun, Handbook of Mathematical Functions,
section 25 <http://mathworld.wolfram.com/FilonsIntegrationFormula.html>`_ or
Allen and Tildesley, *Computer Simulation of Liquids*, Appendix D :cite:`AllTil87`.

Due to the algorithm the number of time samples must be odd i.e. t = dt * [0, ..., 2n]

for some angular frequency w

integral = f(x) @ (+[b/2, g, b, ..., b, g, b/2] * cos(w x)
                   +[-a,  0, 0, ..., 0, 0, a]   * sin(w x))

or

integral =
+ f[0   ] * (b/2 * cos(w x[0]) - a sin(w x[0]))
+ f[1   ] * (g   * cos(w x[1]))
+ f[2   ] * (b   * cos(w x[2]))
+ ...
+ f[2n-2] * (b   * cos(w x[2n-2]))
+ f[2n-1] * (g   * cos(w x[2n-1]))
+ f[2n  ] * (b/2 * cos(w x[2n]) - a sin(w x[2n]))

"""


import numpy as np
import numba
from numpy.typing import NDArray
from typing import Tuple


def fourier_cos_filon(f: NDArray[float],
                      dt: float) -> Tuple[NDArray[float], NDArray[float]]:
    r"""Calculates the direct Fourier cosine transform :math:`F(w)` of a
    function :math:`f(t)` using Filon's integration method.

    Parameters
    ----------
    f
        function values as a 2D array. second axis will be transformed.
        must contain an odd number of elements along second axis.
    dt
        spacing of t-axis (:math:`\Delta t`)

    Returns
    -------
    w
        w containes values in the interval [0, pi/dt).
        length of w is f.shape[1] // 2 + 1.
        These frequencies corresponds to the frequencies from an fft.
        w == 2*np.pi*np.fft.rfftfreq(f.shape[1], dt)
    F
        transform of f along second axis.
        equivalent to np.fft.rfft(f, axis=1).real

    Example
    -------
    A common use case is

    .. code-block:: python

        w, F = fourier_cos_filon(f, dt)
    """

    if f.ndim != 2:
        raise ValueError('f must be 2D and last axis corresponding to integration variable')

    if f.shape[1] % 2 == 0:  # Filon only works for odd N
        raise ValueError('f must contain an odd number of elements along second axis.')

    if f.shape[1] < 2:  # Time signal must be atleast three long
        raise ValueError('f must contain atleast 3 elements along second axis.')

    w = np.linspace(0, 2 * np.pi / (2 * dt), f.shape[1])

    return w, 2 * filon_2D(f, dt)


@numba.njit(fastmath=True, parallel=True)
def filon_2D(f: NDArray[float], dt: float) -> NDArray:
    """Calculates the fourier transform over the last axis using filons method"""

    N_rows, Nt = f.shape

    assert Nt % 2 == 1 and Nt > 1  # Filon only works for odd N

    dw = np.pi / ((Nt - 1) * dt)

    Nw = Nt

    filon_wr = np.zeros((Nw, N_rows), dtype=np.float64)

    for wi in numba.prange(Nw):
        w = wi * dw
        filon_wr[wi] = filon_2D_inner(f, dt, w)

    filon_wr *= dt

    return filon_wr.T.copy()


@numba.njit(fastmath=True)
def filon_2D_inner(f, dt, w):
    """Calculates the transform of 1D f"""

    N_rows, Nt = f.shape

    alpha, beta, gamma = _alpha_beta_gamma_single(dt * w)

    t_arr = np.arange(Nt) * dt

    phase = np.cos(w * t_arr)  # phase = np.sin(w * t_arr)  # for sin transform

    phase[::2] *= beta  # all evens get multiplied with beta
    phase[1:-1:2] *= gamma  # all odds get multiplied with gamma

    # The enpoints (even index) get an extra factor of 1/2
    phase[0] *= 0.5
    phase[-1] *= 0.5

    # The endpoints must also get an extra term
    phase[0] -= alpha * np.sin(w * t_arr[0])  # phase[0] += alpha * np.cos(w * t_arr[0])
    phase[-1] += alpha * np.sin(w * t_arr[-1])  # phase[-1] -= alpha * np.cos(w * t_arr[-1])

    filon = np.zeros(N_rows, dtype=np.float64)

    for r in range(N_rows):
        for t in range(Nt):
            filon[r] += f[r, t] * phase[t]

    return filon


@numba.njit(fastmath=False)
def _alpha_beta_gamma_single(t: float):
    # From theta (t), calculate alpha, beta, and gamma

    if t == 0:
        alpha, beta, gamma = 0.0, 2/3, 4/3
    else:
        alpha = (t**2 + t * np.sin(t) * np.cos(t) - 2 * np.sin(t)**2) / t**3
        beta = 2 * (t * (1 + np.cos(t)**2) - 2 * np.sin(t) * np.cos(t)) / t**3
        gamma = 4 * (np.sin(t) - t * np.cos(t)) / t**3

    return alpha, beta, gamma
