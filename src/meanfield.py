"""Mean-field (population) models.

mpr                    Montbrio-Pazo-Roxin firing-rate equations (exact for QIF networks)
mpr_saddle_node_curve  boundary of the bistable region (MPR 2015, Fig. 1)
kuramoto_daido         Fourier-mode hierarchy of noisy active rotators (Buendia et al. 2021)
ott_antonsen_ar        two-variable (R, Psi) reduction of the same hierarchy
"""
import numpy as np
from numba import njit


# ---------------------------------------------------------------- Montbrio-Pazo-Roxin
@njit(cache=True)
def _mpr_run(I, dt, r0, v0, delta, eta_bar, J, noise_v, seed):
    np.random.seed(seed)
    n = I.shape[0]
    r = np.empty(n)
    v = np.empty(n)
    r[0] = r0
    v[0] = v0
    pi = np.pi
    sq = noise_v * np.sqrt(dt)
    for t in range(n - 1):
        # 2nd-order Runge-Kutta (Heun); with noise: stochastic Heun, same increment in both stages
        xi = sq * np.random.standard_normal() if noise_v > 0 else 0.0
        dr1 = delta / pi + 2.0 * r[t] * v[t]
        dv1 = v[t] ** 2 + eta_bar + J * r[t] + I[t] - (pi * r[t]) ** 2
        rp = r[t] + dt * dr1
        vp = v[t] + dt * dv1 + xi
        dr2 = delta / pi + 2.0 * rp * vp
        dv2 = vp ** 2 + eta_bar + J * rp + I[t + 1] - (pi * rp) ** 2
        r[t + 1] = max(r[t] + 0.5 * dt * (dr1 + dr2), 0.0)
        v[t + 1] = v[t] + 0.5 * dt * (dv1 + dv2) + xi
    return r, v


def mpr(I, dt=1e-3, r0=0.1, v0=-2.0, delta=1.0, eta_bar=-5.0, J=15.0, noise_v=0.0, seed=0):
    """Montbrio-Pazo-Roxin (2015) firing-rate equations (dimensionless, tau = 1):

        dr/dt = delta/pi + 2 r v
        dv/dt = v^2 + eta_bar + J r + I(t) - pi^2 r^2  [+ noise_v * white noise]

    I : input, one value per time step. Returns r(t), v(t).
    """
    return _mpr_run(np.asarray(I, dtype=float), dt, r0, v0, delta, eta_bar, J, float(noise_v), seed)


def mpr_fixed_points(eta_bar, J, delta=1.0, I=0.0):
    """All fixed points (r*, v*) of the MPR equations.

    From dr/dt = 0: v = -delta/(2 pi r). Substituting into dv/dt = 0 gives
    a polynomial in r:  pi^2 r^4 - J r^3 - (eta_bar + I) r^2 - delta^2/(4 pi^2) = 0.
    """
    roots = np.roots([np.pi ** 2, -J, -(eta_bar + I), 0.0, -delta ** 2 / (4 * np.pi ** 2)])
    r = np.sort(roots[(np.abs(roots.imag) < 1e-9) & (roots.real > 0)].real)
    return np.column_stack([r, -delta / (2 * np.pi * r)])


def mpr_jacobian(r, v, J):
    """Jacobian of the MPR equations at (r, v)."""
    return np.array([[2 * v, 2 * r], [J - 2 * np.pi ** 2 * r, 2 * v]])


def mpr_branches(I_values, eta_bar=-5.0, J=15.0, delta=1.0):
    """Fixed points for each input I, with stability.

    Returns a list of (I, r*, v*, stable, focus): stable = both eigenvalues have negative
    real part; focus = the eigenvalues are complex (damped oscillations around the point).
    """
    out = []
    for I in np.atleast_1d(I_values):
        for r, v in mpr_fixed_points(eta_bar, J, delta, I):
            lam = np.linalg.eigvals(mpr_jacobian(r, v, J))
            out.append((I, r, v, bool(np.all(lam.real < 0)), bool(np.any(np.abs(lam.imag) > 1e-12))))
    return out


def lorentzian_v_density(v, v_center, r):
    """Voltage distribution predicted by the mean field: Lorentzian centred at v with half-width pi r."""
    w = np.pi * r
    return (w / np.pi) / ((v - v_center) ** 2 + w ** 2)


def mpr_saddle_node_curve(delta=1.0, r=None):
    """Saddle-node boundary of the bistable region (MPR 2015, Eq. 14), parametrised by r.

    Returns (eta_bar, J) along the curve.
    """
    if r is None:
        r = np.logspace(-3, 1, 2000)
    eta = -np.pi ** 2 * r ** 2 - 3 * delta ** 2 / (2 * np.pi * r) ** 2
    J = 2 * np.pi ** 2 * r + delta ** 2 / (2 * np.pi ** 2 * r ** 3)
    return eta, J


# ---------------------------------------------------------------- active rotators
@njit(cache=True)
def _kd_run(K, omega, gamma, a, J, sigma, dt, n_steps, rec_every, Zinit):
    # Z[k] = <exp(i k phi)>, k = 0..K (Z[0] = 1, Z[k] = 0 for k > K)
    Z = np.zeros(K + 2, dtype=np.complex128)
    Z[:K + 1] = Zinit[:K + 1]
    Z[0] = 1.0
    n_out = n_steps // rec_every
    out = np.zeros((n_out, K + 1), dtype=np.complex128)
    dZ = np.zeros(K + 2, dtype=np.complex128)
    Zp = np.zeros(K + 2, dtype=np.complex128)
    dZ2 = np.zeros(K + 2, dtype=np.complex128)
    for t in range(n_steps):
        if t % rec_every == 0 and t // rec_every < n_out:
            out[t // rec_every] = Z[:K + 1]
        # Heun step
        _kd_rhs(Z, dZ, K, omega, gamma, a, J, sigma)
        Zp[:] = Z + dt * dZ
        Zp[0] = 1.0
        _kd_rhs(Zp, dZ2, K, omega, gamma, a, J, sigma)
        for k in range(1, K + 1):
            Z[k] += 0.5 * dt * (dZ[k] + dZ2[k])
    return out


@njit(cache=True)
def _kd_rhs(Z, dZ, K, omega, gamma, a, J, sigma):
    z1 = Z[1]
    for k in range(1, K + 1):
        dZ[k] = (Z[k] * (1j * omega * k - gamma * k - 0.5 * k * k * sigma * sigma)
                 + 0.5 * a * k * (Z[k + 1] - Z[k - 1])
                 + 0.5 * J * k * (z1 * Z[k - 1] - np.conj(z1) * Z[k + 1]))


def kuramoto_daido(omega=1.0, a=1.0, J=1.0, sigma=0.5, T=200.0, dt=1e-3, rec_dt=0.1,
                   K=50, Z0=0.1 + 0.0j, Zk0=None, gamma=0.0, return_modes=False):
    """Exact mean field of the infinite all-to-all noisy active-rotator network,
    truncated at K Fourier modes Z_k = <exp(i k phi)> (Buendia et al. 2021):

        dZ_k/dt = Z_k (i omega k - gamma k - k^2 sigma^2 / 2) + (a k/2)(Z_{k+1} - Z_{k-1})
                  + (J k/2)(Z_1 Z_{k-1} - conj(Z_1) Z_{k+1})

    gamma : half-width of a Lorentzian spread of natural frequencies (0 = identical units).
    Initial condition: Zk0 (array of modes 0..K, e.g. measured from network phases), or Z_k = Z0^k.
    Returns t and Z_1(t) = R e^{i Psi}; with return_modes=True, all modes (time x K+1).
    """
    rec_every = max(1, int(round(rec_dt / dt)))
    n_steps = int(round(T / dt))
    if Zk0 is None:
        Zk0 = complex(Z0) ** np.arange(K + 1)
    Zk0 = np.asarray(Zk0, dtype=np.complex128)
    Zinit = np.zeros(K + 1, dtype=np.complex128)
    Zinit[:min(K + 1, Zk0.size)] = Zk0[:K + 1]
    out = _kd_run(K, omega, gamma, a, J, sigma, dt, n_steps, rec_every, Zinit)
    t = np.arange(out.shape[0]) * rec_every * dt
    return (t, out) if return_modes else (t, out[:, 1])


def kd_setup(sigma):
    """Number of modes K and time step for kuramoto_daido: more modes for weak noise
    (sharp distributions), smaller steps for strong noise (fast-decaying modes)."""
    s = max(sigma, 1e-3)
    K = int(np.clip(6 / s, 15, 60))
    return K, min(2e-3, 1.0 / (0.5 * K * K * s * s))


def kd_state(a, sigma, J=1.0, omega=1.0, Z0=0.05 + 0j, T=250.0):
    """Long-time state of the exact (Kuramoto-Daido) mean field from initial condition Z_k = Z0^k.
    Returns (mean firing rate, oscillation amplitude = rms of Z around its mean, mean R)."""
    K, dt = kd_setup(sigma)
    t, Zk = kuramoto_daido(omega=omega, a=a, J=J, sigma=sigma, T=T, K=K, dt=dt, Z0=Z0, return_modes=True)
    m = t > T / 2
    Z1 = Zk[m, 1]
    rate = rotator_rate(Zk[m], omega=omega, a=a, J=J, sigma=sigma).mean()
    return rate, np.sqrt(np.mean(np.abs(Z1 - Z1.mean()) ** 2)), np.abs(Z1).mean()


def phase_modes(phases, K):
    """Fourier modes Z_k = mean_j exp(i k phi_j), k = 0..K, of a set of phases."""
    k = np.arange(K + 1)
    return np.exp(1j * np.outer(np.asarray(phases) , k)).mean(axis=0)


def phase_density(Zk, phi):
    """Phase distribution rho(phi) rebuilt from its Fourier modes Z_0..Z_K:
        rho = (1/2pi) [1 + 2 Re sum_{k>=1} Z_k e^{-i k phi}]"""
    Zk = np.atleast_2d(Zk)
    k = np.arange(1, Zk.shape[1])
    e = np.exp(-1j * np.outer(k, np.atleast_1d(phi)))
    rho = (1 + 2 * np.real(Zk[:, 1:] @ e)) / (2 * np.pi)
    return rho if rho.shape[0] > 1 else rho[0]


def rotator_rate(Zk, omega=1.0, a=1.0, J=1.0, sigma=0.5, spike_phase=np.pi / 2):
    """Firing rate = probability flux through the spike phase:
        rate = v(phi_s) rho(phi_s) - (sigma^2/2) d rho/d phi (phi_s),
        v(phi) = omega + a sin(phi) + J Im(Z_1 e^{-i phi})."""
    Zk = np.atleast_2d(Zk)
    k = np.arange(1, Zk.shape[1])
    e = np.exp(-1j * k * spike_phase)
    rho = (1 + 2 * np.real(Zk[:, 1:] @ e)) / (2 * np.pi)
    drho = 2 * np.real(Zk[:, 1:] @ (-1j * k * e)) / (2 * np.pi)
    v = omega + a * np.sin(spike_phase) + J * np.imag(Zk[:, 1] * np.exp(-1j * spike_phase))
    return v * rho - 0.5 * sigma ** 2 * drho


@njit(cache=True)
def _oa_rhs(Z, omega, gamma, a, J, sigma):
    return ((1j * omega - gamma - 0.5 * sigma * sigma) * Z + 0.5 * a * (Z * Z - 1.0)
            + 0.5 * J * Z * (1.0 - (Z.real * Z.real + Z.imag * Z.imag)))


@njit(cache=True)
def _oa_run(omega, gamma, a, J, sigma, dt, n_steps, rec_every, Z0):
    Z = Z0
    n_out = n_steps // rec_every
    out = np.zeros(n_out, dtype=np.complex128)
    for t in range(n_steps):
        if t % rec_every == 0 and t // rec_every < n_out:
            out[t // rec_every] = Z
        d1 = _oa_rhs(Z, omega, gamma, a, J, sigma)
        Zp = Z + dt * d1
        Z = Z + 0.5 * dt * (d1 + _oa_rhs(Zp, omega, gamma, a, J, sigma))
    return out


def ott_antonsen_ar(omega=1.0, a=1.0, J=1.0, sigma=0.5, T=200.0, dt=1e-3, rec_dt=0.1,
                    Z0=0.1 + 0.0j, gamma=0.0):
    """Two-variable (Ott-Antonsen) closure Z_k = Z_1^k of the active-rotator hierarchy:

        dZ/dt = (i omega - gamma - sigma^2/2) Z + (a/2)(Z^2 - 1) + (J/2) Z (1 - |Z|^2)

    or, with Z = R e^{i Psi},
        dR/dt   = R/2 [J (1 - R^2) - sigma^2 - 2 gamma] - (a/2)(1 - R^2) cos(Psi)
        dPsi/dt = omega + a (1 + R^2) sin(Psi) / (2 R)

    Exact without noise (sigma = 0, Lorentzian frequencies of half-width gamma);
    an approximation when sigma > 0. This is the local model of the whole-brain criticality
    model of Rabuffo, ..., Dalla Porta (2025).
    """
    rec_every = max(1, int(round(rec_dt / dt)))
    n_steps = int(round(T / dt))
    Z = _oa_run(omega, gamma, a, J, sigma, dt, n_steps, rec_every, complex(Z0))
    return np.arange(len(Z)) * rec_every * dt, Z


@njit(cache=True)
def _oa_noisy_run(a, J, sigma, omega, eta, dt, n_steps, rec_every, R0, Psi0, seed):
    np.random.seed(seed)
    R, P = R0, Psi0
    s = np.sqrt(2.0 * eta * dt)
    n_out = n_steps // rec_every
    out = np.zeros((n_out, 2))
    for t in range(n_steps):
        if t % rec_every == 0 and t // rec_every < n_out:
            out[t // rec_every, 0] = R
            out[t // rec_every, 1] = P
        dR = 0.5 * R * (J * (1.0 - R * R) - sigma * sigma) - 0.5 * a * (1.0 - R * R) * np.cos(P)
        dP = omega + a * (1.0 + R * R) * np.sin(P) / (2.0 * R)
        R = R + dt * dR
        P = P + dt * dP + s * np.random.standard_normal()
        R = min(max(R, 1e-6), 1.0 - 1e-9)
    return out


def ott_antonsen_noisy(a=0.973, J=1.25, sigma=0.6725, omega=1.0, eta=1e-4, T=2000.0, dt=0.05, rec_dt=0.5,
                       R0=0.5, Psi0=0.0, seed=0, with_rate=True):
    """Two-variable active-rotator mean field (coherence R, mean phase Psi) with a small additive noise
    on the mean phase, as in the whole-brain model of Rabuffo et al. (Phys Rev Research, 2026):
        dR/dt   = R/2 [J (1 - R^2) - sigma^2] - (a/2)(1 - R^2) cos(Psi)
        dPsi/dt = omega + a (1 + R^2) sin(Psi) / (2 R) + noise (intensity eta)
    Defaults = the paper's critical regime (a = 0.973; subcritical 0.979, supercritical 0.967).
    Returns t, R, Psi and, if with_rate, the population firing rate (flux through the spike phase,
    computed from the wrapped-Cauchy phase distribution that the two numbers describe)."""
    rec_every = max(1, int(round(rec_dt / dt)))
    out = _oa_noisy_run(a, J, sigma, omega, eta, dt, int(round(T / dt)), rec_every, R0, Psi0, seed)
    t = np.arange(len(out)) * rec_every * dt
    R, Psi = out[:, 0], out[:, 1]
    if not with_rate:
        return t, R, Psi
    Z = R * np.exp(1j * Psi)
    Zk = Z[:, None] ** np.arange(80)[None, :]
    return t, R, Psi, rotator_rate(Zk, omega=omega, a=a, J=J, sigma=sigma)


# ----------------------------------------------------------------------------- Wilson–Cowan E–I pair
WC = dict(wEE=16.0, wEI=12.0, wIE=15.0, wII=3.0, aE=1.3, thE=4.0, aI=2.0, thI=3.7, tauE=1.0, tauI=1.0)


def wc_sigmoid(x, a, th):
    """Wilson–Cowan response curve, shifted so that S(0) = 0 (no input, no activity)."""
    return 1.0 / (1.0 + np.exp(-a * (x - th))) - 1.0 / (1.0 + np.exp(a * th))


def wilson_cowan(P=1.25, Q=0.0, T=100.0, dt=0.01, E0=0.1, I0=0.05, sigma=0.0, seed=0, **params):
    """Excitatory (E) and inhibitory (I) population activity (Wilson & Cowan 1972):

        tau_E dE/dt = -E + S_E(wEE E - wEI I + P)
        tau_I dI/dt = -I + S_I(wIE E - wII I + Q)

    P, Q: external input to E and I; time in units of the membrane time constant (~10 ms).
    Returns t, E, I.
    """
    p = {**WC, **params}
    n = int(round(T / dt)); E = np.empty(n); I = np.empty(n); E[0], I[0] = E0, I0
    rng = np.random.default_rng(seed); s = sigma * np.sqrt(dt)
    for k in range(n - 1):
        dE = (-E[k] + wc_sigmoid(p["wEE"] * E[k] - p["wEI"] * I[k] + P, p["aE"], p["thE"])) / p["tauE"]
        dI = (-I[k] + wc_sigmoid(p["wIE"] * E[k] - p["wII"] * I[k] + Q, p["aI"], p["thI"])) / p["tauI"]
        E[k + 1] = E[k] + dt * dE + s * rng.standard_normal()
        I[k + 1] = I[k] + dt * dI + s * rng.standard_normal()
    return np.arange(n) * dt, E, I
