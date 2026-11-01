"""Functional connectivity (FC) and its dynamics (FCD) from BOLD time series (time x regions)."""
import numpy as np


def fc(y):
    """Static FC: Pearson correlation between all pairs of regions."""
    return np.corrcoef(y.T)


def sliding_fc(y, window, step):
    """FC in sliding windows (window, step in samples). Returns (n_windows, n_pairs): upper triangles."""
    iu = np.triu_indices(y.shape[1], 1)
    starts = np.arange(0, y.shape[0] - window + 1, step)
    return np.array([np.corrcoef(y[s:s + window].T)[iu] for s in starts]), starts


def fcd(y, window, step):
    """FCD matrix: correlation between the FC patterns of every pair of windows."""
    fcs, starts = sliding_fc(y, window, step)
    return np.corrcoef(fcs), starts


def fcd_values(F, min_lag=1):
    """Values of the FCD matrix above the diagonal (excluding windows closer than min_lag),
    the distribution used to compare FCD between model and data."""
    iu = np.triu_indices(F.shape[0], min_lag)
    return F[iu]


def ks_distance(a, b):
    """Kolmogorov-Smirnov distance between two samples (0 = same distribution, 1 = disjoint)."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, x, side="right") / a.size - np.searchsorted(b, x, side="right") / b.size))


def upper(M):
    """Upper triangle of a square matrix (without the diagonal)."""
    return M[np.triu_indices(M.shape[0], 1)]
