"""Helper code for the course "Modeling brain activity from micro to macro".

The notebooks show the equations; this package only contains the numerics.
Modules:
    neurons     single-neuron models (Hodgkin-Huxley, QIF, theta, active rotator)
    networks    spiking networks (QIF network, active-rotator network)
    meanfield   mean-field models (Montbrio-Pazo-Roxin, Kuramoto-Daido / Ott-Antonsen)
    wholebrain  connectome-based models (MPR nodes, active-rotator mean-field nodes)
    bold        Balloon-Windkessel hemodynamic model
    connectome  loading the HCP group connectome (ENIGMA toolbox)
    analysis    FC, dynamic FC, avalanches
    plotting    shared figure style
"""
