import numpy as np
from numpy.typing import NDArray


def acf_position_dho(t: NDArray[float], w0: float, gamma: float, A: float = 1.0):
    r"""
    The damped damped harmonic oscillator (DHO) autocorrelation function for the position.
    The definition of this function can be found in the `dynasor documentation
    <dynasor.materialsmodeling.org/theory.html#damped-harmonic-oscillator-model>_`.

    Parameters
    ----------
    t
        Time, usually as an array.
    w0
        Natural angular frequency of the DHO.
    gamma
        Damping of DHO.
    A
        Amplitude of the DHO.
    """

    t = np.abs(t)

    if 2 * w0 > gamma:  # underdamped
        we = np.sqrt(w0**2 - gamma**2 / 4.0)
        return A * np.exp(-gamma * t / 2.0) * (
            np.cos(we * t) + 0.5 * gamma / we * np.sin(we * t))
    elif 2 * w0 < gamma:  # overdamped
        tau = 2 / gamma
        tau_S = tau / (1 + np.sqrt(1 - (w0 * tau)**2))
        tau_L = tau / (1 - np.sqrt(1 - (w0 * tau)**2))
        return A / (tau_L - tau_S) * (tau_L * np.exp(-t/tau_L) - tau_S * np.exp(-t/tau_S))
    else:
        tau = 2 / gamma
        return A * np.exp(-t/tau) * (1 + t / tau)


def acf_velocity_dho(t: NDArray[float], w0: float, gamma: float, A: float = 1.0):
    r"""
    The damped damped harmonic oscillator (DHO) autocorrelation function for the velocity.
    The definition of this function can be found in the `dynasor documentation
    <dynasor.materialsmodeling.org/theory.html#damped-harmonic-oscillator-model>_`.

    Parameters
    ----------
    t
        Time, usually as an array.
    w0
        Natural angular frequency of the DHO.
    gamma
        Damping of DHO.
    A
        Amplitude of the DHO.
    """

    t = np.abs(t)

    if 2 * w0 > gamma:  # underdamped
        we = np.sqrt(w0**2 - gamma**2 / 4.0)
        return A * w0**2 * np.exp(-gamma * t / 2.0) * (
            np.cos(we * t) - 0.5 * gamma / we * np.sin(we * t))
    elif 2 * w0 < gamma:  # overdamped
        tau = 2 / gamma
        tau_S = tau / (1 + np.sqrt(1 - (w0 * tau)**2))
        tau_L = tau / (1 - np.sqrt(1 - (w0 * tau)**2))
        return A / (tau_L - tau_S) * (np.exp(-t/tau_S)/tau_S - np.exp(-t/tau_L)/tau_L)
    else:
        tau = 2 / gamma
        return A * np.exp(-t/tau) * (1 - t / tau)


def psd_position_dho(w: NDArray[float], w0: float, gamma: float, A: float = 1.0):
    r"""
    The power spectral density (PSD) function for the damped harmonic oscillator (DHO)
    (i.e., the Fourier transform of the autocorrelation function) for the position.

    The definition of this function can be found in the `dynasor documentation
    <dynasor.materialsmodeling.org/theory.html#damped-harmonic-oscillator-model>_`.

    Parameters
    ----------
    w
        Angular frequency, usually as an array.
    w0
        Natural angular frequency of the DHO.
    gamma
        Damping of DHO.
    A
        Amplitude of the DHO.
    """
    return 2 * w0**2 * A * gamma / ((w**2 - w0**2)**2 + (w * gamma)**2)


def psd_velocity_dho(w: NDArray[float], w0: float, gamma: float, A: float = 1.0):
    r"""
    The power spectral density (PSD) function for the damped harmonic oscillator (DHO)
    (i.e., the Fourier transform of the autocorrelation function) for the position.

    The definition of this function can be found in the `dynasor documentation
    <dynasor.materialsmodeling.org/theory.html#damped-harmonic-oscillator-model>_`.

    Parameters
    ----------
    w
        Angular frequency, usually as an array.
    w0
        Natural angular frequency of the DHO.
    gamma
        Damping of DHO.
    A
        Amplitude of the DHO.
    """
    return w**2 * psd_position_dho(w, w0, gamma, A)
