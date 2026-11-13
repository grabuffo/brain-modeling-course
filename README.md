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
| **Day 1 · Ch. 3** | 05 · A crowd of neurons described by two numbers (Montbrió–Pazó–Roxin) | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C3_NB05_mpr_mean_field.ipynb) |
| | 06 · A crowd of rotators: regimes, bursts, and a two-number model | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D1_C3_NB06_active_rotator_mean_field.ipynb) |
| **Day 2 · Ch. 4** | 07 · Criticality and chaos: small causes, big effects | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C4_NB07_criticality_chaos.ipynb) |
|  | 08 · Bifurcations inside the models: Hopf, saddle-node, SNIC | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C4_NB08_bifurcations.ipynb) |
| **Day 2 · Ch. 5** | 09 · What brain activity looks like: FC, dynamic FC, sleep EEG, avalanches | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB09_brain_activity_data.ipynb) |
|  | 10 · A first whole-brain model (hands-on): Hopf regions on the human connectome | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB10_first_whole_brain_model.ipynb) |
|  | 11 · Bistable regions: neural activity, BOLD and dynamic FC | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB11_bistable_whole_brain.ipynb) |
|  | 12 · The brain's wiring: a network, and gradients | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB12_brain_wiring.ipynb) |
|  | 13 · Reproducing the resting brain: the best fit sits near the bifurcation | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB13_reproducing_rest.ipynb) |
|  | 13b · Not all regions are the same: heterogeneity from the myelin map | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB13b_heterogeneous_regions.ipynb) |
|  | 14a · Silencing a hub (exercise) | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB14a_silencing_a_hub.ipynb) |
|  | 14b · How a local stimulation spreads | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB14b_stimulation_spread.ipynb) |
|  | 14c · A seizure in a bistable brain, and stimulation to stop it | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C5_NB14c_seizure_and_stimulation.ipynb) |
| **Day 2 · Ch. 6** | 15 · Artificial neural networks as surrogate brains | [open](https://colab.research.google.com/github/grabuffo/brain-modeling-course/blob/main/notebooks/D2_C6_NB15_ann_surrogate_brains.ipynb) |

**On your laptop instead:** `git clone https://github.com/grabuffo/brain-modeling-course`, then
`pip install -r requirements.txt` and `jupyter lab notebooks/`.

## Data
- Human connectome (HCP group SC/FC): ENIGMA Toolbox, Larivière et al. 2021, *Nat. Methods* (BSD-3).
- Mouse connectome: Allen Mouse Brain Connectivity, as distributed with The Virtual Brain.
- EEG: PhysioNet EEG Motor Movement/Imagery, Schalk et al. 2004 (ODC-By 1.0); sleep EEG: PhysioNet Sleep-EDF, Kemp et al. 2000 (ODC-By 1.0).
- fMRI time series: OpenNeuro ds000228, Richardson et al. 2018 (CC0), 10 adults, movie watching, Schaefer-100.
- Single human neuron: Allen Cell Types Database. Human single units: DANDI:000004 (CC-BY 4.0).
