import numpy as np
from gwbench import network

############################################################################
### User Choices
############################################################################

# choose the desired detectors
network_spec = ['aLIGO_H','aLIGO_L']
# initialize the network with the desired detectors
net = network.Network(network_spec)

# choose the desired waveform
wf_model_name = 'tf2'
# pass the chosen waveform to the network for initialization
net.set_wf_vars(wf_model_name=wf_model_name)

# pick the desired frequency range
f = np.arange(5.,61.5,2**-4)

# set the injection parameters
inj_params = {
    'Mc':    30.9,
    'eta':   0.247,
    'chi1z': 0,
    'chi2z': 0,
    'DL':    475,
    'tc':    0,
    'phic':  1.9385764842785942,
    'iota':  2.1581020633082915,
    'ra':    6.223352859347546,
    'dec':   -0.5153043355101219,
    'psi':   2.871221497023955,
    'gmst0': 0
    }

# assign with respect to which parameters to take derivatives
deriv_symbs_string = 'Mc eta'

# assign which parameters to convert to cos or log versions
#conv_cos = ('iota','dec')
conv_log = ('Mc, eta')

# choose whether to take Earth's rotation into account
use_rot = 0

# pass all these variables to the network
net.set_net_vars(
    f=f, inj_params=inj_params,
    deriv_symbs_string=deriv_symbs_string, conv_log=conv_log,
    use_rot=use_rot
    )

############################################################################
### GW benchmarking
############################################################################

# compute the WF polarizations
net.calc_wf_polarizations()
# compute the WF polarizations and their derivatives
net.calc_wf_polarizations_derivs_num()

# setup antenna patterns, location phase factors, and PSDs
net.setup_ant_pat_lpf_psds()

# compute the detector responses
net.calc_det_responses()
# compute the detector responses and their derivatives
net.calc_det_responses_derivs_num()

# calculate the network and detector SNRs
net.calc_snrs()

# calculate the network and detector Fisher matrices, condition numbers,
# covariance matrices, error estimates, and inversion errors
net.calc_errors()

# calculate the 90%-credible sky area (in deg)
net.calc_sky_area_90()

############################################################################
### Print results
############################################################################

# print the contents of the detector objects (inside the network)
net.print_detectors()

# print the contents of the network objects
net.print_network()
