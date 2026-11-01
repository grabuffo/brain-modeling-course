"""Connectome-based whole-brain models.

mpr_network : Montbrio-Pazo-Roxin node at every region, coupled through firing rates
              (as in Rabuffo et al. 2021 eNeuro; TVB's MontbrioPazoRoxin model).

    tau dr_i/dt = delta/(pi tau) + 2 r_i v_i
    tau dv_i/dt = v_i^2 + eta + J tau r_i - (pi tau r_i)^2 + I_i(t) + G tau sum_j W_ij r_j + noise

Time in ms, tau = 1 ms by default (TVB convention). Additive noise on r and v.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _mpr_drift(r, v, W, G, eta, J, delta, tau, I, dr, dv):
    N = r.shape[0]
    pi = np.pi
    for i in range(N):
        c = 0.0                                  # network input  sum_j W_ij r_j
        for j in range(N):
            c += W[i, j] * r[j]
        dr[i] = (delta / (pi * tau) + 2.0 * r[i] * v[i]) / tau
        dv[i] = (v[i] * v[i] + eta[i] + J * tau * r[i] - (pi * tau * r[i]) ** 2
                 + I[i] + G * c) / tau


@njit(cache=True)
def _mpr_net_run(W, G, eta, J, delta, tau, sig_r, sig_v, dt, n_steps, rec_every, seed,
                 r0, v0, stim):
    """Stochastic Heun scheme (as in TVB), with adaptive sub-steps.

    When a region jumps from the down to the up state it can go through a very sharp population
    spike (v and r grow by orders of magnitude in a fraction of a ms). The step is then split into
    n_sub sub-steps so that dt_sub * max(|v|, pi tau r) / tau stays below 0.1; the noise
    increment of the full step is split accordingly. Ordinary steps are unchanged.
    stim: (n_steps_stim, N) input; zero after it ends.
    """
    np.random.seed(seed)
    N = W.shape[0]
    r = r0.copy(); v = v0.copy()
    n_out = n_steps // rec_every
    r_out = np.zeros((n_out, N), dtype=np.float32)
    v_out = np.zeros((n_out, N), dtype=np.float32)
    acc_r = np.zeros(N); acc_v = np.zeros(N)
    dr1 = np.empty(N); dv1 = np.empty(N); dr2 = np.empty(N); dv2 = np.empty(N)
    rp = np.empty(N); vp = np.empty(N); nr = np.empty(N); nv = np.empty(N)
    n_stim = stim.shape[0]
    zero = np.zeros(N)
    for t in range(n_steps):
        I = stim[t] if t < n_stim else zero
        fast = 0.0                               # fastest local rate, sets the number of sub-steps
        for i in range(N):
            f = max(abs(v[i]), np.pi * tau * r[i]) / tau
            if f > fast:
                fast = f
        n_sub = max(1, int(np.ceil(fast * dt / 0.1)))
        h = dt / n_sub
        sr = np.sqrt(2.0 * sig_r * h)            # TVB additive-noise convention
        sv = np.sqrt(2.0 * sig_v * h)
        for s in range(n_sub):
            _mpr_drift(r, v, W, G, eta, J, delta, tau, I, dr1, dv1)
            for i in range(N):
                nr[i] = sr * np.random.standard_normal()
                nv[i] = sv * np.random.standard_normal()
                rp[i] = max(r[i] + h * dr1[i] + nr[i], 0.0)
                vp[i] = v[i] + h * dv1[i] + nv[i]
            _mpr_drift(rp, vp, W, G, eta, J, delta, tau, I, dr2, dv2)
            for i in range(N):
                r[i] = max(r[i] + 0.5 * h * (dr1[i] + dr2[i]) + nr[i], 0.0)
                v[i] = v[i] + 0.5 * h * (dv1[i] + dv2[i]) + nv[i]
        for i in range(N):
            acc_r[i] += r[i]; acc_v[i] += v[i]
        if (t + 1) % rec_every == 0:
            o = (t + 1) // rec_every - 1
            for i in range(N):
                r_out[o, i] = acc_r[i] / rec_every
                v_out[o, i] = acc_v[i] / rec_every
                acc_r[i] = 0.0; acc_v[i] = 0.0
    return r_out, v_out


def mpr_network(W, G, T=60_000.0, dt=0.01, rec_dt=1.0, eta=-4.6, J=14.5, delta=0.7, tau=1.0,
                noise_r=0.025, noise_v=0.05, seed=0, r0=None, v0=None, stim=None):
    """Simulate MPR nodes on connectome W with global coupling G.

    Defaults: bistable (down/up) node of Rabuffo et al. 2021 eNeuro.
    T, dt, rec_dt in ms. Steps are subdivided automatically during sharp up-transitions
    (without this, dt = 0.01 blows up at strong coupling on the HCP SC).
    stim: optional external input, array (time steps, regions); zero after it ends.
    Returns t [ms], r (time x regions) [kHz], v (time x regions), averaged over rec_dt.
    """
    N = W.shape[0]
    rng = np.random.default_rng(seed)
    eta = np.full(N, float(eta)) if np.isscalar(eta) else np.asarray(eta, dtype=float)
    r0 = rng.uniform(0.0, 0.2, N) if r0 is None else np.asarray(r0, dtype=float)
    v0 = rng.uniform(-2.0, -1.0, N) if v0 is None else np.asarray(v0, dtype=float)
    stim = np.zeros((1, N)) if stim is None else np.ascontiguousarray(stim, dtype=float)
    rec_every = max(1, int(round(rec_dt / dt)))
    n_steps = int(round(T / dt))
    r, v = _mpr_net_run(np.ascontiguousarray(W, dtype=np.float64), float(G), eta, J, delta, tau,
                        noise_r, noise_v, dt, n_steps, rec_every, seed, r0, v0, stim)
    t = (np.arange(r.shape[0]) + 1) * rec_dt
    return t, r, v
