"""Single-neuron models.

All integrators are explicit and written out so students can read them.
Units: Hodgkin-Huxley in mV, ms, uA/cm^2; QIF, theta and active rotator are dimensionless.
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- Hodgkin-Huxley
# Classic squid-axon parameters, resting potential shifted to -65 mV
# (as in Dayan & Abbott 2001, Gerstner et al. 2014).
HH = dict(C=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.387)


@njit(cache=True)
def _hh_rates(V):
    am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def hh_rates(V):
    """Opening (alpha) and closing (beta) rates of the m, h, n gates [1/ms]."""
    return _hh_rates(np.asarray(V, dtype=float))


@njit(cache=True)
def _hh_run(I, dt, C, gNa, gK, gL, ENa, EK, EL, V0, gE, gI, EE, EI):
    n_steps = I.shape[0]
    V = np.empty(n_steps); m = np.empty(n_steps); h = np.empty(n_steps); n = np.empty(n_steps)
    V[0] = V0
    am, bm, ah, bh, an, bn = _hh_rates(V0)
    m[0] = am / (am + bm); h[0] = ah / (ah + bh); n[0] = an / (an + bn)   # gates at rest
    for t in range(n_steps - 1):
        am, bm, ah, bh, an, bn = _hh_rates(V[t])
        INa = gNa * m[t] ** 3 * h[t] * (V[t] - ENa)
        IK = gK * n[t] ** 4 * (V[t] - EK)
        IL = gL * (V[t] - EL)
        Isyn = -gE[t] * (V[t] - EE) - gI[t] * (V[t] - EI)        # synaptic input (zero unless given)
        V[t + 1] = V[t] + dt * (I[t] + Isyn - INa - IK - IL) / C
        # exponential Euler for the gates (exact for fixed V, always stays in [0, 1])
        m_inf = am / (am + bm); tm = 1.0 / (am + bm)
        h_inf = ah / (ah + bh); th = 1.0 / (ah + bh)
        n_inf = an / (an + bn); tn = 1.0 / (an + bn)
        m[t + 1] = m_inf + (m[t] - m_inf) * np.exp(-dt / tm)
        h[t + 1] = h_inf + (h[t] - h_inf) * np.exp(-dt / th)
        n[t + 1] = n_inf + (n[t] - n_inf) * np.exp(-dt / tn)
    return V, m, h, n


def hodgkin_huxley(I, dt=0.01, V0=-65.0, **params):
    """Simulate the Hodgkin-Huxley neuron.

    I : array of injected current [uA/cm^2], one value per time step.
    Returns V, m, h, n (arrays of the same length as I).
    """
    p = {**HH, **params}
    I = np.asarray(I, dtype=float); zero = np.zeros_like(I)
    return _hh_run(I, dt, p["C"], p["gNa"], p["gK"], p["gL"], p["ENa"], p["EK"], p["EL"], V0, zero, zero, 0.0, -75.0)


def hodgkin_huxley_syn(g_exc, g_inh, dt=0.01, I=None, E_exc=0.0, E_inh=-75.0, V0=-65.0, **params):
    """HH neuron receiving synaptic input as conductances [mS/cm^2], one value per time step.

    Excitatory synapses (glutamate, AMPA) pull V towards E_exc ~ 0 mV; inhibitory synapses (GABA_A, chloride)
    pull V towards E_inh ~ -75 mV:  I_syn = -g_exc (V - E_exc) - g_inh (V - E_inh).
    Returns V, m, h, n.
    """
    p = {**HH, **params}
    g_exc = np.asarray(g_exc, dtype=float); g_inh = np.asarray(g_inh, dtype=float)
    I = np.zeros_like(g_exc) if I is None else np.asarray(I, dtype=float)
    return _hh_run(I, dt, p["C"], p["gNa"], p["gK"], p["gL"], p["ENa"], p["EK"], p["EL"], V0, g_exc, g_inh, E_exc, E_inh)


def synaptic_conductance(spike_times, T, dt=0.01, g_max=0.1, tau=3.0):
    """Conductance trace [mS/cm^2] from presynaptic spikes at spike_times [ms]: a jump of g_max, then decay with tau."""
    n = int(round(T / dt)); g = np.zeros(n); kick = np.zeros(n)
    for ts in spike_times:
        i = int(round(ts / dt))
        if 0 <= i < n: kick[i] += g_max
    decay = np.exp(-dt / tau)
    for i in range(n):
        g[i] = (g[i - 1] * decay if i else 0.0) + kick[i]
    return g


def spike_times(V, dt, threshold=0.0):
    """Times of upward threshold crossings."""
    idx = np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    return idx * dt


def hh_fi_curve(currents, T=500.0, dt=0.01, discard=100.0):
    """Firing rate [Hz] of the HH neuron for each constant current."""
    rates = []
    for I0 in currents:
        V, *_ = hodgkin_huxley(np.full(int(T / dt), I0), dt)
        st = spike_times(V, dt)
        st = st[st > discard]
        rates.append(1000.0 * len(st) / (T - discard))
    return np.array(rates)


# ---------------------------------------------------------------- QIF neuron
@njit(cache=True)
def _qif_run(I, dt, v0, v_peak):
    n_steps = I.shape[0]
    v = np.empty(n_steps)
    v[0] = v0
    spikes = []
    for t in range(n_steps - 1):
        v_new = v[t] + dt * (v[t] * v[t] + I[t])
        if v_new >= v_peak:          # spike: reset to -v_peak
            spikes.append(t * dt)
            v_new = -v_peak
        v[t + 1] = v_new
    return v, np.array(spikes)


def qif(I, dt=1e-3, v0=-1.0, v_peak=100.0):
    """Quadratic integrate-and-fire neuron  dv/dt = v^2 + I.

    The true model resets at v = +inf to v = -inf; numerically we use +-v_peak.
    Returns v (array) and spike times.
    """
    return _qif_run(np.asarray(I, dtype=float), dt, v0, v_peak)


def qif_rate(I):
    """Analytical firing rate of the QIF neuron: sqrt(I)/pi for I > 0, else 0."""
    I = np.asarray(I, dtype=float)
    return np.where(I > 0, np.sqrt(np.clip(I, 0, None)) / np.pi, 0.0)


# ---------------------------------------------------------------- phase models
@njit(cache=True)
def _phase_run(omega, a, I, sigma, dt, phi0, seed, theta_neuron):
    np.random.seed(seed)
    n_steps = I.shape[0]
    phi = np.empty(n_steps)
    phi[0] = phi0
    sq = sigma * np.sqrt(dt)
    for t in range(n_steps - 1):
        if theta_neuron:   # d theta/dt = (1 - cos theta) + (1 + cos theta) I
            f = (1.0 - np.cos(phi[t])) + (1.0 + np.cos(phi[t])) * I[t]
        else:              # d phi/dt = omega + a sin(phi) + I
            f = omega + a * np.sin(phi[t]) + I[t]
        phi[t + 1] = phi[t] + dt * f + sq * np.random.standard_normal()
    return phi


def theta_neuron(I, dt=1e-3, theta0=-np.pi / 2, sigma=0.0, seed=0):
    """Ermentrout-Kopell theta neuron. Equivalent to the QIF through v = tan(theta/2)."""
    return _phase_run(0.0, 0.0, np.asarray(I, dtype=float), sigma, dt, theta0, seed, True)


def active_rotator(omega=1.0, a=0.0, I=None, T=100.0, dt=1e-3, phi0=0.0, sigma=0.0, seed=0):
    """Active rotator (Shinomoto-Kuramoto):  d phi/dt = omega + a sin(phi) + I + noise.

    a < omega : the unit oscillates (rotates around the circle).
    a > omega : the unit is excitable (rests at a stable fixed point).
    a = omega : SNIC bifurcation.
    Returns the unwrapped phase.
    """
    n = int(round(T / dt))
    I = np.zeros(n) if I is None else np.asarray(I, dtype=float)
    return _phase_run(omega, a, I, sigma, dt, phi0, seed, False)


def phase_spikes(phi, dt, spike_phase=np.pi / 2):
    """Spike = the phase crosses spike_phase (mod 2 pi) going forward.

    The running maximum of phi is used, so that noise jittering back and forth
    across spike_phase counts as a single spike.
    """
    k = np.floor((np.maximum.accumulate(phi) - spike_phase) / (2 * np.pi))
    return np.where(np.diff(k) > 0)[0] * dt


# ---------------------------------------------------------------- I_Na,p + I_K (2D, type I)
# Persistent sodium + potassium model, low-threshold version with a SNIC bifurcation
# at I ~ 4.51 (Izhikevich 2007, ch. 4). Units: mV, ms, uA/cm^2.
INAPK = dict(C=1.0, gL=8.0, EL=-80.0, gNa=20.0, ENa=60.0, gK=10.0, EK=-90.0, tau_n=1.0)


def inapk_minf(V):
    return 1.0 / (1.0 + np.exp((-20.0 - V) / 15.0))


def inapk_ninf(V):
    return 1.0 / (1.0 + np.exp((-25.0 - V) / 5.0))


@njit(cache=True)
def _inapk_run(I, dt, V0, n0, C, gL, EL, gNa, ENa, gK, EK, tau_n):
    N = I.shape[0]
    V = np.empty(N); n = np.empty(N)
    V[0] = V0; n[0] = n0
    for t in range(N - 1):
        minf = 1.0 / (1.0 + np.exp((-20.0 - V[t]) / 15.0))
        ninf = 1.0 / (1.0 + np.exp((-25.0 - V[t]) / 5.0))
        dV = (I[t] - gL * (V[t] - EL) - gNa * minf * (V[t] - ENa) - gK * n[t] * (V[t] - EK)) / C
        V[t + 1] = V[t] + dt * dV
        n[t + 1] = n[t] + dt * (ninf - n[t]) / tau_n
    return V, n


def inapk(I, dt=0.01, V0=-65.0, n0=None, **params):
    """I_Na,p + I_K model:
        C dV/dt = I - gL (V-EL) - gNa m_inf(V) (V-ENa) - gK n (V-EK)
        dn/dt   = (n_inf(V) - n) / tau_n
    Returns V, n.
    """
    p = {**INAPK, **params}
    n0 = inapk_ninf(V0) if n0 is None else n0
    return _inapk_run(np.asarray(I, dtype=float), dt, V0, n0, p["C"], p["gL"], p["EL"],
                      p["gNa"], p["ENa"], p["gK"], p["EK"], p["tau_n"])


def inapk_nullclines(V, I, **params):
    """V-nullcline (N-shaped) and n-nullcline (sigmoid) as functions of V."""
    p = {**INAPK, **params}
    nV = (I - p["gL"] * (V - p["EL"]) - p["gNa"] * inapk_minf(V) * (V - p["ENa"])) / (p["gK"] * (V - p["EK"]))
    return nV, inapk_ninf(V)


def inapk_steady_current(V, **params):
    """Steady-state I-V curve: the current needed to hold the neuron at rest at voltage V.
    Fixed points of the model are the solutions of I = I_ss(V)."""
    p = {**INAPK, **params}
    return (p["gL"] * (V - p["EL"]) + p["gNa"] * inapk_minf(V) * (V - p["ENa"])
            + p["gK"] * inapk_ninf(V) * (V - p["EK"]))


def hh_steady_current(V):
    """Steady-state I-V curve of the Hodgkin-Huxley model (all gates at their x_inf)."""
    am, bm, ah, bh, an, bn = hh_rates(V)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    p = HH
    return (p["gNa"] * m**3 * h * (V - p["ENa"]) + p["gK"] * n**4 * (V - p["EK"]) + p["gL"] * (V - p["EL"]))


def inapk_fixed_points(I, V=np.linspace(-90, 20, 22001), **params):
    """Fixed points (V*, n*) of the I_Na,p + I_K model: solutions of I = I_ss(V)."""
    g = inapk_steady_current(V, **params) - I
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    Vs = V[idx] - g[idx] * (V[idx + 1] - V[idx]) / (g[idx + 1] - g[idx])   # linear interpolation
    return np.column_stack([Vs, inapk_ninf(Vs)])
