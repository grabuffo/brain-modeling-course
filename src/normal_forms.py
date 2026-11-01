"""Normal forms of the bifurcations met in the course (Chapter 4).

hopf      dz/dt = (mu + i omega) z + c |z|^2 z + d |z|^4 z + I(t) + noise
          supercritical: c = -1, d = 0;  subcritical: c = +1, d = -1
cusp      dx/dt = mu + x - x^3 + I(t) + noise     (two saddle-nodes, bistable switch)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _hopf_run(mu, omega, c, d, sigma, dt, z0, I, seed):
    np.random.seed(seed)
    n = I.shape[0]
    z = np.empty(n, dtype=np.complex128)
    z[0] = z0
    sq = sigma * np.sqrt(dt)
    for t in range(n - 1):
        zz = z[t]
        r2 = zz.real * zz.real + zz.imag * zz.imag
        f = (mu[t] + 1j * omega) * zz + c * r2 * zz + d * r2 * r2 * zz + I[t]
        z[t + 1] = zz + dt * f + sq * (np.random.standard_normal() + 1j * np.random.standard_normal())
    return z


def hopf(mu, omega=1.0, kind="super", sigma=0.0, T=100.0, dt=1e-3, z0=0.1 + 0j, I=None, seed=0):
    """Hopf normal form (complex z = x + i y), Euler-Maruyama.

    kind = 'super': dz/dt = (mu + i omega) z - |z|^2 z
    kind = 'sub'  : dz/dt = (mu + i omega) z + |z|^2 z - |z|^4 z
    mu can be a number or an array (one value per step, for slow ramps).
    sigma: independent white noise on x and y.  I: complex input, one value per step.
    Returns t, z.
    """
    n = int(round(T / dt))
    I = np.zeros(n, dtype=np.complex128) if I is None else np.asarray(I, dtype=np.complex128)
    c, d = (-1.0, 0.0) if kind == "super" else (1.0, -1.0)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n,)).copy()
    z = _hopf_run(mu, omega, c, d, sigma, dt, complex(z0), I, seed)
    return np.arange(n) * dt, z


def hopf_cycles(mu, kind="super"):
    """Radii of the limit cycles: (stable, unstable) lists."""
    if kind == "super":
        return ([np.sqrt(mu)] if mu > 0 else []), []
    disc = 1 + 4 * mu                      # r^2 = (1 +- sqrt(1 + 4 mu)) / 2
    if disc < 0:
        return [], []
    big, small = (1 + np.sqrt(disc)) / 2, (1 - np.sqrt(disc)) / 2
    stable = [np.sqrt(big)]
    unstable = [np.sqrt(small)] if small > 0 else []
    return stable, unstable


@njit(cache=True)
def _cusp_run(mu, sigma, dt, x0, I, seed):
    np.random.seed(seed)
    n = I.shape[0]
    x = np.empty(n)
    x[0] = x0
    sq = sigma * np.sqrt(dt)
    for t in range(n - 1):
        x[t + 1] = x[t] + dt * (mu + x[t] - x[t] ** 3 + I[t]) + sq * np.random.standard_normal()
    return x


def cusp(mu, sigma=0.0, T=100.0, dt=1e-3, x0=-1.0, I=None, seed=0):
    """Bistable switch dx/dt = mu + x - x^3 + I(t) + sigma xi. Saddle-nodes at mu = +-2/(3 sqrt 3)."""
    n = int(round(T / dt))
    I = np.zeros(n) if I is None else np.asarray(I, dtype=float)
    return np.arange(n) * dt, _cusp_run(mu, sigma, dt, x0, I, seed)


def cusp_fixed_points(mu):
    """Real roots of mu + x - x^3 = 0, with stability (slope 1 - 3x^2 < 0)."""
    r = np.roots([-1.0, 0.0, 1.0, mu])
    x = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return x, (1 - 3 * x ** 2) < 0


def autocorr_time(x, dt):
    """Correlation time: first lag at which the autocorrelation falls below 1/e.

    For a complex signal z (e.g. the Hopf normal form) the modulus of the complex
    autocorrelation is used, so the oscillation itself does not count: |C(tau)| = e^{mu tau}
    for the linear damped oscillator. Returns (tau, autocorrelation)."""
    x = np.asarray(x)
    x = x - x.mean()
    n = x.size
    f = np.fft.fft(x, 2 * n)
    ac = np.fft.ifft(f * np.conj(f))[:n]
    ac = np.abs(ac) if np.iscomplexobj(x) else ac.real
    ac = ac / ac[0]
    below = np.where(ac < np.exp(-1))[0]
    return (below[0] if below.size else n) * dt, ac
