#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import bilby
import numpy as np
from bilby.gw.conversion import generate_all_bbh_parameters

import waveform_dataset_3p as waveform_dataset
from bilby.gw.utils import calculate_time_to_merger
from bilby.gw.prior import CBCPriorDict


class Bilby_Posterior:

    def __init__(self, wg, model_dir, three_parameter=True):
        self._wg = wg
        self._outdir = model_dir+"bilby_plots"
        self._three_parameter = three_parameter

    def find_result(self, idx, params_true):
        # Set the duration and sampling frequency of the data segment that we're
        # going to inject the signal into
        # duration = self._wg.duration
        duration = 8.0
        sampling_frequency = self._wg.sampling_freq
        minimum_frequency = self._wg.fmin

        # Specify the output directory and the name of the simulation.
        label = str(idx)
        bilby.core.utils.setup_logger(outdir=self._outdir, label=label)

        # Set up a random seed for result reproducibility.  This is optional!
        np.random.seed(88170235)
        geocent_time = 1126259642.413

        if self._three_parameter:
            mass_1 = float(params_true[self._wg.INTRINSIC_PARAMS['mass1']])
            mass_2 = float(params_true[self._wg.INTRINSIC_PARAMS['mass2']])
            luminosity_distance = float(params_true[self._wg.EXTRINSIC_PARAMS['distance']])

            a_1 = float(self._wg.other_params[[self._wg.OTHER_PARAMS['a1']]])
            a_2 = float(self._wg.other_params[[self._wg.OTHER_PARAMS['a2']]])
            tilt_1 = float(self._wg.other_params[[self._wg.OTHER_PARAMS['theta1']]])
            tilt_2 = float(self._wg.other_params[[self._wg.OTHER_PARAMS['theta2']]])
            phi_12 = float(self._wg.other_params[[self._wg.OTHER_PARAMS['phi_12']]])
            phi_jl = float(self._wg.other_params[[self._wg.OTHER_PARAMS['phi_JL']]])
            theta_jn = float(self._wg.other_params[[self._wg.OTHER_PARAMS['theta_JN']]])

            psi = float(self._wg.other_params[[self._wg.OTHER_PARAMS['pol_angle']]])
            phase = float(self._wg.other_params[[self._wg.OTHER_PARAMS['phase']]])
            ra = float(self._wg.other_params[self._wg.OTHER_PARAMS['right_ascension']])
            dec = float(self._wg.other_params[self._wg.OTHER_PARAMS['declination']])
        else:
            mass_1 = float(params_true[self._wg.INTRINSIC_PARAMS['mass1']])
            mass_2 = float(params_true[self._wg.INTRINSIC_PARAMS['mass2']])
            luminosity_distance = float(params_true[self._wg.EXTRINSIC_PARAMS['distance']])

            a_1 = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['a1']]])
            a_2 = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['a2']]])
            tilt_1 = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta1']]])
            tilt_2 = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta2']]])
            phi_12 = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['phi_12']]])
            phi_jl = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['phi_JL']]])
            theta_jn = float(self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta_JN']]])

            psi = float(self._wg.other_params[[self._wg.EXTRINSIC_PARAMS['pol_angle']]])
            phase = float(self._wg.other_params[[self._wg.EXTRINSIC_PARAMS['phase']]])
            ra = float(self._wg.other_params[self._wg.EXTRINSIC_PARAMS['right_ascension']])
            dec = float(self._wg.other_params[self._wg.EXTRINSIC_PARAMS['declination']])

        injection_parameters = dict(
            mass_1=mass_1,
            mass_2=mass_2,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
            phi_jl=phi_jl,
            luminosity_distance=luminosity_distance,
            theta_jn=theta_jn,
            psi=psi,
            phase=phase,
            geocent_time=geocent_time,
            ra=ra,
            dec=dec,
        )

        # Fixed arguments passed into the source model
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=minimum_frequency,
        )

        # Create the waveform_generator using a LAL BinaryBlackHole source function
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

        # Set up interferometers.  In this case we'll use two interferometers
        # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
        # sensitivity
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - 2,
        )
        ifos.inject_signal(
            waveform_generator=waveform_generator, parameters=injection_parameters
        )

        # Set up a PriorDict, which inherits from dict.
        # By default we will sample all terms in the signal models.  However, this will
        # take a long time for the calculation, so for this example we will set almost
        # all of the priors to be equall to their injected values.  This implies the
        # prior is a delta function at the true, injected value.  In reality, the
        # sampler implementation is smart enough to not sample any parameter that has
        # a delta-function prior.
        # The above list does *not* include mass_1, mass_2, theta_jn and luminosity
        # distance, which means those are the parameters that will be included in the
        # sampler.  If we do nothing, then the default priors get used.
        priors = bilby.gw.prior.BBHPriorDict()
        for key in [
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "psi",
            "ra",
            "dec",
            "geocent_time",
            "phase",
        ]:
            priors[key] = injection_parameters[key]

        if self._three_parameter:
            # We can make uniform distributions.
            del priors["chirp_mass"], priors["mass_ratio"]
            # We can make uniform distributions.
            priors["mass_1"] = bilby.core.prior.Uniform(
                name="mass_1", minimum=mass_1-1, maximum=mass_1+1
            )
            priors["mass_2"] = bilby.core.prior.Uniform(
                name="mass_2", minimum=mass_2-1, maximum=mass_2+1
            )
            priors["luminosity_distance"] = bilby.core.prior.Uniform(
                name="luminosity_distance", minimum=luminosity_distance-10, maximum=luminosity_distance+10
            )
        else:
            print('TODO')

        # Perform a check that the prior does not extend to a parameter space longer than the data
        priors.validate_prior(duration, minimum_frequency)

        # Initialise the likelihood by passing in the interferometer data (ifos) and
        # the waveform generator
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator
        )

        # Run sampler.  In this case we're going to use the `dynesty` sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            npoints=1000,
            npool=4,
            injection_parameters=injection_parameters,
            outdir=outdir,
            label=label,
        )

        # Make a corner plot.
        result.plot_corner()


# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = "outdir"

label = "fast_tutorial"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=800.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.

# Set up prior
# This loads in a predefined set of priors for BBHs.
priors = bilby.gw.prior.BBHPriorDict()
# These parameters will not be sampled
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]
# We can make uniform distributions.
del priors["chirp_mass"], priors["mass_ratio"]
# We can make uniform distributions.
priors["mass_1"] = bilby.core.prior.Uniform(
    name="mass_1", minimum=35, maximum=37
)
priors["mass_2"] = bilby.core.prior.Uniform(
    name="mass_2", minimum=28, maximum=30
)
priors["luminosity_distance"] = bilby.core.prior.Uniform(
    name="luminosity_distance", minimum=790, maximum=810
)

# N = 100
# samples = priors.sample(N)
# samples = generate_all_bbh_parameters(samples)
#
# print(samples)
#
# durations = np.array([
#
#     calculate_time_to_merger(
#         frequency=minimum_frequency,
#         mass_1=mass_11,
#         mass_2=mass_22,
#     )
#
#     for (mass_11, mass_22) in zip(samples["mass_1"], samples["mass_2"])])
#
# longest_duration = max(durations)
# print('Longest Durationnnn')
# print(longest_duration)

print(priors)

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency, warning=True)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Make a corner plot.
result.plot_corner()
