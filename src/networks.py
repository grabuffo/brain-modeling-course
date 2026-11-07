"""Spiking networks.

qif_network            all-to-all QIF network of Montbrio, Pazo & Roxin (2015)
active_rotator_network all-to-all noisy active rotators (Buendia et al. 2021)
rotator_sheet          the same rotators on a 2D sheet, coupled to their neighbours within a radius R
"""
import numpy as np
from numba import njit


def lorentzian_etas(N, eta_bar, delta):
    """Deterministic Lorentzian sample of the excitabilities eta_j (MPR 2015, App. A).

    eta_j = eta_bar + delta * tan(pi/2 * (2j - N - 1)/(N + 1)),  j = 1..N
    """
    j = np.arange(1, N + 1)
    return eta_bar + delta * np.tan(0.5 * np.pi * (2 * j - N - 1) / (N + 1))


@njit(cache=True)
def _qif_net_run(eta, J, I, dt, v_peak, tau_s, n_rec, rec_every, v0, snap_steps):
    N = eta.shape[0]
    n_steps = I.shape[0]
    v = v0.copy()
    refr = np.zeros(N)                       # remaining refractory time
    t_refr = 2.0 / v_peak                    # time spent between +v_peak and -v_peak
    n_buf = max(1, int(round(tau_s / dt)))   # rectangular synaptic window
    buf = np.zeros(n_buf)
    buf_sum = 0.0
    n_out = n_steps // rec_every
    r_out = np.zeros(n_out)
    v_out = np.zeros(n_out)
    spk_t = []
    spk_i = []
    acc_spikes = 0.0
    snaps = np.full((snap_steps.shape[0], N), np.nan)
    k_snap = 0
    for t in range(n_steps):
        if k_snap < snap_steps.shape[0] and t == snap_steps[k_snap]:
            for i in range(N):
                if refr[i] <= 0.0:          # neurons in the middle of a spike are left out (NaN)
                    snaps[k_snap, i] = v[i]
            k_snap += 1
        s = buf_sum / (N * n_buf * dt)       # population rate seen by the synapses
        n_sp = 0
        for i in range(N):
            if refr[i] > 0.0:
                refr[i] -= dt
                if refr[i] <= 0.0:
                    v[i] = -v_peak
                continue
            v[i] += dt * (v[i] * v[i] + eta[i] + J * s + I[t])
            if v[i] >= v_peak:
                n_sp += 1
                refr[i] = t_refr
                v[i] = v_peak
                if i < n_rec:
                    spk_t.append(t * dt)
                    spk_i.append(i)
        k = t % n_buf
        buf_sum += n_sp - buf[k]
        buf[k] = n_sp
        acc_spikes += n_sp
        if (t + 1) % rec_every == 0:
            o = (t + 1) // rec_every - 1
            r_out[o] = acc_spikes / (N * rec_every * dt)
            acc_spikes = 0.0
            # centre of the voltage distribution = median (the Lorentzian has no finite mean;
            # the median is not affected by the few neurons close to the spike cutoff)
            v_out[o] = np.median(v[refr <= 0.0])
    return r_out, v_out, np.array(spk_t), np.array(spk_i), snaps


def qif_network(I, N=10000, eta_bar=-5.0, delta=1.0, J=15.0, dt=1e-4, v_peak=1000.0,
                tau_s=1e-3, bin_size=1e-2, n_rec=300, v0=None, seed=0, snap_times=None, eta=None):
    """All-to-all network of N QIF neurons (MPR 2015, Eqs. 1-2).

        dv_j/dt = v_j^2 + eta_j + J s(t) + I(t)

    s(t) is the population firing rate over the last tau_s time units.
    Spike when v_j reaches v_peak, reset to -v_peak after 2/v_peak (the time an exact QIF spends
    beyond +-v_peak). v_peak = 1000 is needed for the network to match the mean field closely;
    with v_peak = 100 the collective oscillations are damped too slowly.
    I : external input, one value per time step.
    Returns time of the bins, rate r, voltage v (median over neurons), and spikes (times, ids) of the first n_rec neurons.
    eta : optional array of excitabilities (overrides the Lorentzian eta_bar, delta, N).
    snap_times : optional times (from the start of I) at which the voltages of all neurons are saved;
                 if given, the snapshots (len(snap_times) x N, NaN during a spike) are returned last.
    """
    rng = np.random.default_rng(seed)
    if eta is None:
        eta = lorentzian_etas(N, eta_bar, delta)
    else:
        eta = np.array(eta, dtype=float)
        N = eta.size
    rng.shuffle(eta)                         # so that the recorded neurons are a random sample
    if v0 is None:
        v0 = rng.uniform(-2.0, -1.0, N)
    rec_every = max(1, int(round(bin_size / dt)))
    snap_steps = np.sort(np.round(np.atleast_1d(snap_times if snap_times is not None else [])
                                  / dt).astype(np.int64))
    r, v, st, si, snaps = _qif_net_run(eta, J, np.asarray(I, dtype=float), dt, v_peak, tau_s,
                                       n_rec, rec_every, np.asarray(v0, dtype=float), snap_steps)
    t = (np.arange(len(r)) + 1) * rec_every * dt
    if snap_times is not None:
        return t, r, v, st, si, snaps
    return t, r, v, st, si


@njit(cache=True)
def _ar_net_run(N, omega, a, J, sigma, dt, n_steps, rec_every, n_rec, seed, phi0, spike_phase, n_spk):
    np.random.seed(seed)
    phi = phi0.copy()
    sq = sigma * np.sqrt(dt)
    n_out = n_steps // rec_every
    Z = np.zeros(n_out, dtype=np.complex128)
    phases = np.zeros((n_out, n_rec), dtype=np.float32)
    counts = np.zeros(n_steps, dtype=np.int32)      # population spikes in each time step
    f1 = np.empty(N)
    xi = np.empty(N)
    php = np.empty(N)
    # spike = forward crossing of spike_phase (mod 2 pi) by the running maximum of the phase
    k_spk = np.floor((phi - spike_phase) / (2 * np.pi))
    pmax = phi.copy()
    spk_t = []
    spk_i = []
    for t in range(n_steps):
        # Kuramoto order parameter Z = R e^{i Psi} = mean e^{i phi}
        zr = 0.0
        zi = 0.0
        for i in range(N):
            zr += np.cos(phi[i])
            zi += np.sin(phi[i])
        zr /= N
        zi /= N
        if t % rec_every == 0 and t // rec_every < n_out:
            o = t // rec_every
            Z[o] = zr + 1j * zi
            for i in range(n_rec):
                phases[o, i] = phi[i]
        # Heun (predictor-corrector) step;  J R sin(Psi - phi) = J (zi cos phi - zr sin phi)
        for i in range(N):
            f1[i] = omega[i] + a * np.sin(phi[i]) + J * (zi * np.cos(phi[i]) - zr * np.sin(phi[i]))
            xi[i] = sq * np.random.standard_normal()
            php[i] = phi[i] + dt * f1[i] + xi[i]
        zr2 = 0.0
        zi2 = 0.0
        for i in range(N):
            zr2 += np.cos(php[i])
            zi2 += np.sin(php[i])
        zr2 /= N
        zi2 /= N
        for i in range(N):
            f2 = omega[i] + a * np.sin(php[i]) + J * (zi2 * np.cos(php[i]) - zr2 * np.sin(php[i]))
            phi[i] += 0.5 * dt * (f1[i] + f2) + xi[i]
            if phi[i] > pmax[i]:
                pmax[i] = phi[i]
                k_new = np.floor((pmax[i] - spike_phase) / (2 * np.pi))
                if k_new > k_spk[i]:
                    k_spk[i] = k_new
                    counts[t] += 1
                    if i < n_spk:
                        spk_t.append((t + 1) * dt)
                        spk_i.append(i)
    return Z, phases, counts, np.array(spk_t), np.array(spk_i)


def active_rotator_network(N=2000, omega=1.0, a=1.0, J=1.0, sigma=0.5, T=200.0, dt=0.01,
                           rec_dt=0.1, n_rec=200, seed=0, phi0=None, spike_phase=np.pi / 2,
                           return_spikes=False, n_spk=0):
    """All-to-all network of noisy active rotators (Buendia et al. 2021).

        dphi_j/dt = omega + a sin(phi_j) + (J/N) sum_k sin(phi_k - phi_j) + sigma xi_j(t)

    With a = 0 this is the noisy Kuramoto model with identical frequencies.
    omega can also be an array (one natural frequency per unit).
    Returns t, the order parameter Z(t) = R e^{i Psi}, and the phases of n_rec units.
    With return_spikes=True, also the number of population spikes in every time step dt
    and the spike times / ids of the first n_spk units (spike = forward crossing of spike_phase).
    """
    rng = np.random.default_rng(seed)
    if phi0 is None:
        phi0 = rng.uniform(0, 2 * np.pi, N)
    rec_every = max(1, int(round(rec_dt / dt)))
    n_steps = int(round(T / dt))
    omega = np.broadcast_to(np.asarray(omega, dtype=float), (N,)).copy()
    Z, ph, counts, st, si = _ar_net_run(N, omega, a, J, sigma, dt, n_steps, rec_every, n_rec, seed,
                                        np.asarray(phi0, dtype=float), spike_phase, n_spk)
    t = np.arange(len(Z)) * rec_every * dt
    if return_spikes:
        return t, Z, ph, counts, st, si
    return t, Z, ph


def phase_raster(t, phases, spike_phase=np.pi / 2):
    """Spike times/ids from recorded (unwrapped) phases: forward crossings of spike_phase
    (running maximum, so that noisy back-and-forth crossings count once)."""
    k = np.floor((np.maximum.accumulate(phases, axis=0) - spike_phase) / (2 * np.pi))
    jumps = np.diff(k, axis=0) > 0
    ti, ii = np.nonzero(jumps)
    return t[ti + 1], ii


def rotator_sheet(L=100, omega=1.0, a=1.01, J=2.0, sigma=0.3, R=1, T=200.0, dt=0.02, rec_dt=0.5,
                  seed=0, phi0=None, spike_phase=np.pi / 2):
    """Active rotators on an L x L sheet (periodic borders), each coupled to its neighbours.

        dphi_j/dt = omega + a sin(phi_j) + J * mean_{k near j} sin(phi_k - phi_j) + sigma xi_j(t)

    "Near j" = the square of side 2R+1 centred on j. R=None means all-to-all (the whole sheet):
    exactly the network of active_rotator_network.
    As in the all-to-all case, the coupling only needs the LOCAL order parameter
    Z_j = C_j + i S_j = mean_{k near j} e^{i phi_k}:   mean sin(phi_k - phi_j) = S_j cos(phi_j) - C_j sin(phi_j).

    Returns t (n_rec,), phases (n_rec, L, L), rate (n_rec,) = fraction of units that spiked per unit time
    in each recording interval, and the final phases (to continue a simulation).
    """
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(seed)
    if phi0 is None:                                   # start everyone at rest (or at phase 0 if oscillatory)
        phi0 = np.full((L, L), np.pi + np.arcsin(omega / a) if a > omega else 0.0)
    phi = np.array(phi0, dtype=float)
    every = max(1, int(round(rec_dt / dt)))
    n_steps = int(round(T / dt))
    t_rec, ph_rec, rate = [], [], []
    top = phi.copy()                                   # running maximum: count each spike once despite noise
    n_spk = 0
    for step in range(n_steps):
        c, s = np.cos(phi), np.sin(phi)
        if R is None:                                  # all-to-all: one global order parameter
            C, S = c.mean(), s.mean()
        else:                                          # local order parameter: average over the square
            C = uniform_filter(c, size=2 * R + 1, mode="wrap")
            S = uniform_filter(s, size=2 * R + 1, mode="wrap")
        coupling = J * (S * c - C * s)
        phi = phi + dt * (omega + a * s + coupling) + sigma * np.sqrt(dt) * rng.standard_normal(phi.shape)
        k_old = np.floor((top - spike_phase) / (2 * np.pi))
        top = np.maximum(top, phi)
        n_spk += np.count_nonzero(np.floor((top - spike_phase) / (2 * np.pi)) > k_old)
        if (step + 1) % every == 0:
            t_rec.append((step + 1) * dt)
            ph_rec.append(np.mod(phi, 2 * np.pi).astype(np.float32))
            rate.append(n_spk / phi.size / (every * dt))
            n_spk = 0
    return np.array(t_rec), np.array(ph_rec), np.array(rate), phi
