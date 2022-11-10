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
from fisher_info import Fisher
from numpy.random import default_rng
import scipy.ndimage
import scipy.stats
from scipy import optimize

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
            self._ifos[i].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=self._wg.freqs,
                                                                                          psd_array=self._wg.psd)

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

        #signal = np.fft.irfft(np.pad(strains[0]*self._wg.psd[self._wg.fft_mask]**(-0.5), (1, 0),
        #            'constant')) * self._wg.df * self._wg.length / np.sqrt(self._wg.bandwidth)
        #td_axis = np.arange(0, self._wg.duration, step=self._wg.dt)

        #signal = np.fft.irfft(noise)

        #plt.figure()
        #plt.plot(np.abs(strains[0]))
        #plt.xscale('log')
        #plt.yscale('log')
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
                name="mass_1", minimum=mass_1-5., maximum=mass_1+5.
            )
            priors["mass_2"] = bilby.core.prior.Uniform(
                name="mass_2", minimum=mass_2-5., maximum=mass_2+5.
            )
            priors["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
                name="luminosity_distance", minimum=luminosity_distance-100.,
                maximum=luminosity_distance+100.
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
            npoints=100,
            npool=8,
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


class HellingerDistance:

    def __init__(self, model_dir, N, waveform_generator):

        self._outdir = model_dir + "hellinger_cache"
        self.N = N
        self.distances = np.zeros(N)
        self.wg = waveform_generator
        self.fisher = Fisher(self.wg)
        self.bilby = Bilby_Posterior(self.wg, model_dir)

    def distance_mean(self):

        return np.mean(self.distances)

    def distance_std(self):

        return np.std(self.distances)

    def get_distances_array(self):

        return self.distances

    def calculate_hellinger_distances(self, save=False):

        if self.wg.extrinsic_at_train:
            print('dataset shouldn\'t be extrinsic at train')
            return

        rng = default_rng()
        idxs = rng.choice(self.wg.dataset_len, size=self.N, replace=False)

        for i in range(0, self.N):

            params_true = self.wg.params[idxs[i]]
            print('True Params:')
            print(params_true)

            cov_matrix = self.fisher.compute_analytical_cov_m1_m2(params=params_true)
            print('Covariance matrix:')
            print(cov_matrix)

            bilby_result, _ = self.bilby.find_result(idxs[i], params_true)

            print(bilby_result.kde)

            kde = bilby_result.kde
            samples = bilby_result.samples

            # H, edges = np.histogramdd(bilby_result.samples, bins=20,density=True)
            # print(H.shape)
            #
            # bins = scipy.ndimage.gaussian_filter(H, sigma=0.5)
            #
            # max_index = np.asarray(np.unravel_index(bins.argmax(), H.shape))
            #
            # print(max_index)
            #
            # max_likelihood_pt = np.zeros(3)
            # for m in range(0, len(max_index)):
            #     max_likelihood_pt[m] = (edges[m][max_index[m]] + edges[m][max_index[m]+1]) / 2.
            #
            # print(max_likelihood_pt)
            #

            opt = optimize.minimize(lambda x: -kde(x), x0=params_true)

            print('Max likelihood point')
            print(opt.x)

            max_likelihood_pt = opt.x

            fisher_density = scipy.stats.multivariate_normal.pdf(bilby_result.samples,
                                                                 mean=max_likelihood_pt,
                                                                 cov=cov_matrix)

            dx = 0

            for s in range(0, samples.shape[0]):
                self.distances[i] += 1./2*((np.sqrt(fisher_density[s]) -
                                                  np.sqrt(kde.pdf(samples[s])))**2)
                dx += kde.pdf(samples[s])
                #print(self.distances[i])

            dx = 1/dx
            print('dx')
            print(dx)
            self.distances[i] = np.sqrt(self.distances[i]*dx)
            print(self.distances[i])

        if save:
            np.savetxt(self._outdir+'true_params.txt', self.wg.params[idxs], delimiter=',')
            np.savetxt(self._outdir+'distances.txt', self.distances, delimiter=',')

        return self.distances


