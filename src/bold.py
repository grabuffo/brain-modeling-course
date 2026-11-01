"""Balloon-Windkessel hemodynamic model (Friston et al. 2000, 2003; Stephan et al. 2007).

Turns neural activity x(t) into a BOLD signal:
    ds/dt = x - kappa s - gamma (f - 1)          vasodilatory signal
    df/dt = s                                    blood inflow
    tau dv/dt = f - v^(1/alpha)                  blood volume
    tau dq/dt = f E(f)/E0 - v^(1/alpha) q / v    deoxyhemoglobin
    BOLD = V0 [k1 (1 - q) + k2 (1 - q/v) + k3 (1 - v)]
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _bw(x, dt, kappa, gamma, tau, alpha, E0, V0, k1, k2, k3):
    n_t, n = x.shape
    y = np.empty((n_t, n))
    for j in prange(n):
        s = 0.0; f = 1.0; v = 1.0; q = 1.0
        for t in range(n_t):
            ds = x[t, j] - kappa * s - gamma * (f - 1.0)
            df = s
            dv = (f - v ** (1.0 / alpha)) / tau
            dq = (f * (1.0 - (1.0 - E0) ** (1.0 / f)) / E0 - v ** (1.0 / alpha) * q / v) / tau
            s += dt * ds; f += dt * df; v += dt * dv; q += dt * dq
            y[t, j] = V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
    return y


def balloon_windkessel(x, dt, kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32, E0=0.34, V0=0.02):
    """x: neural activity, shape (time, regions), sampled every dt SECONDS. Returns BOLD (same shape)."""
    k1, k2, k3 = 7.0 * E0, 2.0, 2.0 * E0 - 0.2
    return _bw(np.ascontiguousarray(x, dtype=np.float64), dt, kappa, gamma, tau, alpha, E0, V0, k1, k2, k3)


def downsample(y, dt, TR):
    """Keep one sample every TR seconds."""
    step = int(round(TR / dt))
    return y[::step]


def bandpass(y, TR, low=0.008, high=0.08, order=2):
    """Detrend, zero-phase Butterworth band-pass and z-score each column (time x regions),
    as for the HCP data used in the course (0.008-0.08 Hz)."""
    from scipy.signal import butter, filtfilt, detrend
    b, a = butter(order, [low * 2 * TR, high * 2 * TR], btype="bandpass")
    yf = filtfilt(b, a, detrend(y, axis=0), axis=0)
    return (yf - yf.mean(0)) / yf.std(0)
