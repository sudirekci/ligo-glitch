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

import matplotlib.pyplot as plt

class Bilby_Posterior:

    def __init__(self, wg, model_dir, three_parameter=True):

        self._wg = wg
        self._outdir = model_dir + "bilby_plots"
        self._three_parameter = three_parameter

        #self._wg.no_detectors = 1

        self._ifos = []
        val_list = list(self._wg.DETECTORS.values())
        key_list = list(self._wg.DETECTORS.keys())

        for i in range(self._wg.no_detectors):
            det_name = key_list[val_list.index(self._wg.detector_list[i])]
            self._ifos.append(bilby.gw.detector.get_empty_interferometer(det_name))

        waveform_arguments = dict(
            waveform_approximant=self._wg.approximant,
            reference_frequency=50.0,
            minimum_frequency=self._wg.fmin,
        )

        self._waveform_generator = bilby.gw.WaveformGenerator(
            duration=self._wg.duration,
            sampling_frequency=self._wg.sampling_freq,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

    def find_result(self, idx, params_true):

        # Specify the output directory and the name of the simulation.
        label = str(idx)
        bilby.core.utils.setup_logger(outdir=self._outdir, label=label)

        # Set up a random seed for result reproducibility.  This is optional!
        np.random.seed(88170235)
        geocent_time = self._wg.duration

        strains = []

        if self._wg.domain == 'TD':
            print('This method only works in the frequency domain.')
            return

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
            tc = float(self._wg.other_params[self._wg.OTHER_PARAMS['tc']])


        else:

            mass_1 = float(params_true[self._wg.INTRINSIC_PARAMS['mass1']])
            mass_2 = float(params_true[self._wg.INTRINSIC_PARAMS['mass2']])
            luminosity_distance = float(params_true[self._wg.EXTRINSIC_PARAMS['distance']])

            a_1 = float(params_true[[self._wg.INTRINSIC_PARAMS['a1']]])
            a_2 = float(params_true[[self._wg.INTRINSIC_PARAMS['a2']]])
            tilt_1 = float(params_true[[self._wg.INTRINSIC_PARAMS['theta1']]])
            tilt_2 = float(params_true[[self._wg.INTRINSIC_PARAMS['theta2']]])
            phi_12 = float(params_true[[self._wg.INTRINSIC_PARAMS['phi_12']]])
            phi_jl = float(params_true[[self._wg.INTRINSIC_PARAMS['phi_JL']]])
            theta_jn = float(params_true[[self._wg.INTRINSIC_PARAMS['theta_JN']]])

            psi = float(params_true[[self._wg.EXTRINSIC_PARAMS['pol_angle']]])
            phase = float(params_true[[self._wg.EXTRINSIC_PARAMS['phase']]])
            ra = float(params_true[self._wg.EXTRINSIC_PARAMS['right_ascension']])
            dec = float(params_true[self._wg.EXTRINSIC_PARAMS['declination']])
            tc = float(params_true[self._wg.OTHER_PARAMS['tc']])

        if self._wg.add_glitch:
            print('TODO')

        # compute the polarizations
        hp, hc = self._wg.compute_hp_hc(-1, params=params_true)

        # divide by the distance

        hp = hp / luminosity_distance
        hc = hc / luminosity_distance

        for j in range(0, self._wg.no_detectors):

            det = self._wg.detectors[j]

            fp, fc = det.antenna_pattern(ra, dec, psi, geocent_time)

            dt = det.time_delay_from_earth_center(ra, dec, geocent_time)
            timeshift = dt + tc
            print('dt', dt)
            print('tc', tc)
            print('timeshift', timeshift)

            noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(self._wg.length))) \
                        [self._wg.fft_mask] * self._wg.dt * \
                    np.sqrt(self._wg.bandwidth*self._wg.psd[self._wg.fft_mask])

            # self._wg.bandwidth

            strains.append((fp * hp + fc * hc) *
                           np.exp(-1j * 2 * np.pi *
                                  self._wg.freqs[self._wg.fft_mask] * timeshift)+noise)

        #print(noise)
        #print('*************************')
        #print(np.pad(noise, (1, 0),'constant'))

        signal = np.fft.irfft(np.pad(strains[0]*self._wg.psd[self._wg.fft_mask]**(-0.5), (1, 0),
                    'constant')) * self._wg.df * self._wg.length / np.sqrt(self._wg.bandwidth)
        td_axis = np.arange(0, self._wg.duration, step=self._wg.dt)

        #signal = np.fft.irfft(noise)

        plt.figure()
        plt.plot(np.abs(strains[0]))
        plt.xscale('log')
        plt.yscale('log')
        #plt.plot(self._wg.psd)
        #plt.show()

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

        priors = bilby.gw.prior.BBHPriorDict()
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

        if self._three_parameter:

            del priors["chirp_mass"], priors["mass_ratio"]
            # We can make uniform distributions.
            priors["mass_1"] = bilby.core.prior.Uniform(
                name="mass_1", minimum=mass_1*0.75, maximum=mass_1*1.25
            )
            priors["mass_2"] = bilby.core.prior.Uniform(
                name="mass_2", minimum=mass_2*0.75, maximum=mass_2*1.25
            )
            priors["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
                name="luminosity_distance", minimum=luminosity_distance*0.75,
                maximum=luminosity_distance*1.25
            )
            priors["mass_ratio"] = bilby.core.prior.Constraint(minimum=1./2, maximum=1.)
        else:
            print('TODO')

        # set the strain data directly
        for j in range(0, self._wg.no_detectors):

            #strains[j] = np.concatenate((np.conjugate(strains[j]), np.asarray([0], dtype=complex),
            #                             strains[j]))

            #np.insert(strains[j], 0, 0)

            self._ifos[j].set_strain_data_from_frequency_domain_strain(np.insert(strains[j], 0, 0),
                                                                       frequency_array=self._wg.freqs,
                                                                       #start_time=geocent_time-self._wg.duration+2.)
                                                                       )

            # start_time = geocent_time-duration+2

        print('Strain data set')

        #plt.figure()
        #plt.plot(np.fft.irfft(self._ifos[1].whitened_frequency_domain_strain)\
        #         * self._wg.df * self._wg.length / np.sqrt(self._wg.bandwidth))
        #plt.show()

        # Initialise the likelihood by passing in the interferometer data (ifos) and
        # the waveform generator
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=self._ifos, waveform_generator=self._waveform_generator
        )

        print('Likelihood defined')

        # Run sampler.  In this case we're going to use the `dynesty` sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            npoints=1000,
            npool=4,
            injection_parameters=injection_parameters,
            outdir=self._outdir,
            label=label,
            #sample="slice",
            #slices=3,
            #nlive=500,
        )

        # Make a corner plot.
        fig = result.plot_corner()

        return result, fig



    # def __init__(self, wg, model_dir, three_parameter=True):
    #     self._wg = wg
    #     self._outdir = model_dir+"bilby_plots"
    #     self._three_parameter = three_parameter
    #
    # def find_result(self, idx, params_true):
    #     # Set the duration and sampling frequency of the data segment that we're
    #     # going to inject the signal into
    #     # duration = self._wg.duration
    #     duration = 8.0
    #     sampling_frequency = self._wg.sampling_freq
    #     minimum_frequency = self._wg.fmin
    #
    #     # Specify the output directory and the name of the simulation.
    #     label = str(idx)
    #     bilby.core.utils.setup_logger(outdir=self._outdir, label=label)
    #
    #     # Set up a random seed for result reproducibility.  This is optional!
    #     np.random.seed(88170235)
    #     geocent_time = 1126259642.413
    #
    #     if self._three_parameter:
    #         mass_1 = params_true[self._wg.INTRINSIC_PARAMS['mass1']])
    #         mass_2 = params_true[self._wg.INTRINSIC_PARAMS['mass2']])
    #         luminosity_distance = params_true[self._wg.EXTRINSIC_PARAMS['distance']])
    #
    #         a_1 = self._wg.other_params[[self._wg.OTHER_PARAMS['a1']]])
    #         a_2 = self._wg.other_params[[self._wg.OTHER_PARAMS['a2']]])
    #         tilt_1 = self._wg.other_params[[self._wg.OTHER_PARAMS['theta1']]])
    #         tilt_2 = self._wg.other_params[[self._wg.OTHER_PARAMS['theta2']]])
    #         phi_12 = self._wg.other_params[[self._wg.OTHER_PARAMS['phi_12']]])
    #         phi_jl = self._wg.other_params[[self._wg.OTHER_PARAMS['phi_JL']]])
    #         theta_jn = self._wg.other_params[[self._wg.OTHER_PARAMS['theta_JN']]])
    #
    #         psi = self._wg.other_params[[self._wg.OTHER_PARAMS['pol_angle']]])
    #         phase = self._wg.other_params[[self._wg.OTHER_PARAMS['phase']]])
    #         ra = self._wg.other_params[self._wg.OTHER_PARAMS['right_ascension']])
    #         dec = self._wg.other_params[self._wg.OTHER_PARAMS['declination']])
    #     else:
    #         mass_1 = params_true[self._wg.INTRINSIC_PARAMS['mass1']])
    #         mass_2 = params_true[self._wg.INTRINSIC_PARAMS['mass2']])
    #         luminosity_distance = params_true[self._wg.EXTRINSIC_PARAMS['distance']])
    #
    #         a_1 = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['a1']]])
    #         a_2 = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['a2']]])
    #         tilt_1 = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta1']]])
    #         tilt_2 = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta2']]])
    #         phi_12 = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['phi_12']]])
    #         phi_jl = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['phi_JL']]])
    #         theta_jn = self._wg.other_params[[self._wg.INTRINSIC_PARAMS['theta_JN']]])
    #
    #         psi = self._wg.other_params[[self._wg.EXTRINSIC_PARAMS['pol_angle']]])
    #         phase = self._wg.other_params[[self._wg.EXTRINSIC_PARAMS['phase']]])
    #         ra = self._wg.other_params[self._wg.EXTRINSIC_PARAMS['right_ascension']])
    #         dec = self._wg.other_params[self._wg.EXTRINSIC_PARAMS['declination']])
    #
    #     injection_parameters = dict(
    #         mass_1=mass_1,
    #         mass_2=mass_2,
    #         a_1=a_1,
    #         a_2=a_2,
    #         tilt_1=tilt_1,
    #         tilt_2=tilt_2,
    #         phi_12=phi_12,
    #         phi_jl=phi_jl,
    #         luminosity_distance=luminosity_distance,
    #         theta_jn=theta_jn,
    #         psi=psi,
    #         phase=phase,
    #         geocent_time=geocent_time,
    #         ra=ra,
    #         dec=dec,
    #     )
    #
    #     # Fixed arguments passed into the source model
    #     waveform_arguments = dict(
    #         waveform_approximant="IMRPhenomPv2",
    #         reference_frequency=50.0,
    #         minimum_frequency=minimum_frequency,
    #     )
    #
    #     # Create the waveform_generator using a LAL BinaryBlackHole source function
    #     waveform_generator = bilby.gw.WaveformGenerator(
    #         duration=duration,
    #         sampling_frequency=sampling_frequency,
    #         frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    #         parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    #         waveform_arguments=waveform_arguments,
    #     )
    #
    #     # Set up interferometers.  In this case we'll use two interferometers
    #     # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    #     # sensitivity
    #     ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    #     ifos.set_strain_data_from_power_spectral_densities(
    #         sampling_frequency=sampling_frequency,
    #         duration=duration,
    #         start_time=injection_parameters["geocent_time"] - 2,
    #     )
    #     ifos.inject_signal(
    #         waveform_generator=waveform_generator, parameters=injection_parameters
    #     )
    #
    #     # Set up a PriorDict, which inherits from dict.
    #     # By default we will sample all terms in the signal models.  However, this will
    #     # take a long time for the calculation, so for this example we will set almost
    #     # all of the priors to be equall to their injected values.  This implies the
    #     # prior is a delta function at the true, injected value.  In reality, the
    #     # sampler implementation is smart enough to not sample any parameter that has
    #     # a delta-function prior.
    #     # The above list does *not* include mass_1, mass_2, theta_jn and luminosity
    #     # distance, which means those are the parameters that will be included in the
    #     # sampler.  If we do nothing, then the default priors get used.
    #     priors = bilby.gw.prior.BBHPriorDict()
    #     for key in [
    #         "a_1",
    #         "a_2",
    #         "tilt_1",
    #         "tilt_2",
    #         "phi_12",
    #         "phi_jl",
    #         "psi",
    #         "ra",
    #         "dec",
    #         "geocent_time",
    #         "phase",
    #     ]:
    #         priors[key] = injection_parameters[key]
    #
    #     if self._three_parameter:
    #         # We can make uniform distributions.
    #         del priors["chirp_mass"], priors["mass_ratio"]
    #         # We can make uniform distributions.
    #         priors["mass_1"] = bilby.core.prior.Uniform(
    #             name="mass_1", minimum=mass_1-1, maximum=mass_1+1
    #         )
    #         priors["mass_2"] = bilby.core.prior.Uniform(
    #             name="mass_2", minimum=mass_2-1, maximum=mass_2+1
    #         )
    #         priors["luminosity_distance"] = bilby.core.prior.Uniform(
    #             name="luminosity_distance", minimum=luminosity_distance-25, maximum=luminosity_distance+25
    #         )
    #     else:
    #         print('TODO')
    #
    #     # Perform a check that the prior does not extend to a parameter space longer than the data
    #     priors.validate_prior(duration, minimum_frequency)
    #
    #     # Initialise the likelihood by passing in the interferometer data (ifos) and
    #     # the waveform generator
    #     likelihood = bilby.gw.GravitationalWaveTransient(
    #         interferometers=ifos, waveform_generator=waveform_generator
    #     )
    #
    #     # Run sampler.  In this case we're going to use the `dynesty` sampler
    #     result = bilby.run_sampler(
    #         likelihood=likelihood,
    #         priors=priors,
    #         sampler="dynesty",
    #         npoints=1000,
    #         npool=4,
    #         injection_parameters=injection_parameters,
    #         outdir=self._outdir,
    #         label=label,
    #     )
    #
    #     # Make a corner plot.
    #     result.plot_corner()
