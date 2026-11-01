"""HCP group connectome (Larivière et al. 2021, ENIGMA Toolbox, BSD-3 license).

Files in data/connectome/ were downloaded from
https://github.com/MICA-MNI/ENIGMA/tree/master/enigmatoolbox/datasets/matrices/hcp_connectivity
They are group averages of HCP young adults: structural connectivity (diffusion tractography,
log-transformed streamline counts) and resting-state functional connectivity (Fisher-z).
"""
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "connectome"
PARCELLATIONS = {"dk68": "ctx", "schaefer100": "ctx_schaefer_100"}


def load_hcp(parcellation="schaefer100", normalize=True, weights="log", homotopic=0.0):
    """Return SC (n x n), empirical group FC (n x n) and region labels.

    weights: "log"    as distributed by ENIGMA (log-transformed fibre counts: dense, homogeneous);
             "counts" undo the log (exp(w) - 1): heavy-tailed, a few strong connections dominate,
                      as in tracer connectomes and raw tractography.
    normalize: True or "rowsum": divide by the largest row sum (every region receives a total input <= 1);
               "max": divide by the largest entry (as in TVB tutorials); False: no normalisation.
    homotopic: add this weight (after normalisation) between mirror regions of the two hemispheres.
               Tractography misses many callosal fibres, while homotopic regions are strongly correlated
               in fMRI; whole-brain models often compensate this way.
    """
    tag = PARCELLATIONS[parcellation]
    sc = np.loadtxt(DATA / f"strucMatrix_{tag}.csv", delimiter=",")
    fc = np.loadtxt(DATA / f"funcMatrix_{tag}.csv", delimiter=",")
    labels = np.loadtxt(DATA / f"strucLabels_{tag}.csv", delimiter=",", dtype=str)
    labels = np.atleast_1d(labels).ravel()
    np.fill_diagonal(sc, 0.0)
    if weights == "counts":
        sc = np.expm1(sc)
    if normalize is True or normalize == "rowsum":
        sc = sc / sc.sum(axis=1).max()
    elif normalize == "max":
        sc = sc / sc.max()
    if homotopic:
        for k, j in homotopic_pairs(parcellation, labels):
            sc[k, j] += homotopic; sc[j, k] += homotopic
    return sc, fc, labels


def homotopic_pairs(parcellation, labels):
    """Pairs (left, right) of mirror regions. Schaefer: the right region closest to the mirror image
    (x -> -x) of each left region's centre; DK-68: same name in the two hemispheres."""
    labels = list(labels)
    cen = DATA / f"centroids_{PARCELLATIONS[parcellation]}.csv"
    if cen.exists():
        xyz = np.loadtxt(cen, delimiter=",")
        left = [k for k, l in enumerate(labels) if "LH" in l]
        right = np.array([k for k, l in enumerate(labels) if "RH" in l])
        pairs = []
        for k in left:
            m = xyz[k] * np.array([-1, 1, 1])
            pairs.append((k, int(right[np.argmin(np.linalg.norm(xyz[right] - m, axis=1))])))
        return pairs
    idx = {l: k for k, l in enumerate(labels)}
    return [(k, idx[l.replace("L_", "R_", 1)]) for k, l in enumerate(labels) if l.startswith("L_") and l.replace("L_", "R_", 1) in idx]


# ----------------------------------------------------------------------------- mouse
MOUSE = Path(__file__).resolve().parents[1] / "data" / "mouse"
MOUSE_CORTEX = np.r_[0:30, 74:104]                         # cortical regions of the Allen-148 parcellation
MOUSE_KEEP = np.delete(np.arange(60), [19, 17, 14, 21, 22, 49, 47, 44, 51, 52])   # 50 regions compared with fMRI


def load_mouse():
    """Allen mouse connectome (148 regions; Oh et al. 2014, Melozzi et al. 2017) as used in
    Rabuffo et al. 2021 (eNeuro) and 2025: tracer-based, directed, normalised by its largest weight.

    Returns SC (148 x 148), labels, region centres (mm), and the indices of 50 cortical regions
    (used for FC; the same selection as Rabuffo et al. 2025).
    """
    sc = np.loadtxt(MOUSE / "weights.txt")
    np.fill_diagonal(sc, 0.0)
    sc = sc / sc.max()
    labels = np.array([l.strip() for l in open(MOUSE / "region_labels.txt")])
    centres = np.loadtxt(MOUSE / "centres.txt").T             # file: 3 rows (x, y, z) x 148 regions
    return sc, labels, centres, MOUSE_CORTEX[MOUSE_KEEP]
