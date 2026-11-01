# Modeling the brain from neurons to networks

Course notebooks: micro → macro brain modeling (2 days × 8 h), Chieti.

**How to use:** click a notebook's *Open in Colab* link. You only need a Google account; nothing to install.
Run the first cell (setup) once: it downloads the course code. To keep your changes, use **File → Save a copy in Drive**.
The first run of each simulation takes 10–30 s extra while the code compiles.

| | Notebook | Colab |
|---|---|---|
| **Before the course** | 00b · Equations of change: derivatives, differential equations, and how a computer solves them | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C0_NB00b_equations_primer.ipynb) |
| **Day 1 · Ch. 0** | 00 · A tour of human brain data, from one neuron to the whole brain | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C0_NB00_data_tour.ipynb) |
| **Day 1 · Ch. 1** | 01 · The Hodgkin–Huxley neuron | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C1_NB01_hodgkin_huxley.ipynb) |
| | 02 · One variable is enough: QIF, theta neuron, active rotator | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C1_NB02_qif_theta_rotator.ipynb) |
| **Day 1 · Ch. 2** | 03 · A network of QIF neurons | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C2_NB03_qif_network.ipynb) |
| | 04 · A network of active rotators: synchronisation | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C2_NB04_active_rotator_network.ipynb) |
| **Day 1 · Ch. 3** | 05 · A population in two equations (Montbrió–Pazó–Roxin) | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C3_NB05_mpr_mean_field.ipynb) |
| | 06 · Mean field of the active rotators: exact, and approximate | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C3_NB06_active_rotator_mean_field.ipynb) |
| **Day 2 · Ch. 4** | 07 · A bifurcation toolkit | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C4_NB07_bifurcation_toolkit.ipynb) |
| | 08 · Three roads to synchrony, and where the brain-like avalanches are | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C4_NB08_hybrid_transitions_avalanches.ipynb) |
| **Day 2 · Ch. 5** | 09 · A whole-brain model | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB09_whole_brain_model.ipynb) |
| | 10 · A virtual lesion: a local change, a whole-brain effect | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB10_perturbing_the_model.ipynb) |
| **Day 2 · Ch. 6** | 11 · Neural networks as surrogate brains | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C6_NB11_neural_network_surrogates.ipynb) |
| **Appendix** | 12 · Which parameters explain the data? Sweeps, degeneracy and simulation-based inference | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_App_NB12_parameter_inference.ipynb) |

**On your laptop instead:** `git clone https://github.com/grabuffo/brain-modeling-course`, then
`pip install -r requirements.txt` and `jupyter lab notebooks/`.

## Data
- Human connectome (HCP group SC/FC): ENIGMA Toolbox, Larivière et al. 2021, *Nat. Methods* (BSD-3).
- Mouse connectome: Allen Mouse Brain Connectivity, as distributed with The Virtual Brain.
- EEG: PhysioNet EEG Motor Movement/Imagery, Schalk et al. 2004 (ODC-By 1.0).
- Single human neuron: Allen Cell Types Database. Human single units: DANDI:000004 (CC-BY 4.0).
