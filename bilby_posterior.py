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
import corner
import os

import waveform_dataset_3p as waveform_dataset
from bilby.gw.utils import calculate_time_to_merger
from bilby.gw.prior import CBCPriorDict

import matplotlib.pyplot as plt

class Bilby_Posterior:


    # DON'T FORGET TO FIX REF TIME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

    def find_result(self, idx, params_true, label=None, glitch_array=None,
                 glitch_start_time=0, glitch_domain='TD'):

        # glitch_start_time the start of the array in seconds (NOT THE MIDDLE)

        # Specify the output directory and the name of the simulation.
        if label is None:
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

        # compute the polarizations
        hp, hc = self._wg.compute_hp_hc(-1, params=params_true)

        # divide by the distance

        hp = hp / luminosity_distance
        hc = hc / luminosity_distance

        timeshifts = np.zeros(self._wg.no_detectors)

        for j in range(0, self._wg.no_detectors):

            det = self._wg.detectors[j]

            fp, fc = det.antenna_pattern(ra, dec, psi, geocent_time)

            dt = det.time_delay_from_earth_center(ra, dec, geocent_time)
            timeshifts[j] = dt + tc


            noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(self._wg.length))) \
                        [self._wg.fft_mask] * self._wg.dt * \
                    np.sqrt(self._wg.bandwidth*self._wg.psd[self._wg.fft_mask])

            # without noise
            # noise = 0.

            # self._wg.bandwidth

            strains.append((fp * hp + fc * hc) *
                           np.exp(-1j * 2 * np.pi *
                                  self._wg.freqs[self._wg.fft_mask] * timeshifts[j])+noise)


        if self._wg.add_glitch or glitch_array is not None:

            if glitch_domain == 'TD':

                if len(glitch_array) != self._wg.length:
                    # pad the glitch
                    # first, left side:
                    glitch_duration = len(glitch_array) / self._wg.length * self._wg.duration
                    left_side = glitch_start_time + self._wg.duration / 2
                    left_pad = np.max(left_side * self._wg.length / self._wg.duration, 0)
                    print(left_pad)

                    right_side = self._wg.duration / 2. - glitch_duration - glitch_start_time
                    right_pad = np.max(right_side * self._wg.length / self._wg.duration, 0)
                    print(right_pad)

                    glitch_array = np.concatenate((np.zeros(int(left_pad)),
                                                   glitch_array, np.zeros(int(right_pad))))

                glitch_array = np.fft.rfft(glitch_array)[self._wg.fft_mask] * np.exp(
                    -1j * 2 * np.pi * self._wg.freqs[self._wg.fft_mask] * glitch_start_time) * \
                    self._wg.dt * np.sqrt(self._wg.bandwidth*self._wg.psd[self._wg.fft_mask])

                for j in range(0, self._wg.no_detectors):

                    strains[j] += glitch_array

            else:

                print('TODO')

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
            npoints=250,
            npool=8,
            injection_parameters=injection_parameters,
            outdir=self._outdir,
            label=label,
            #nact=10,
            #sample="slice",
            #slices=3,
            #nlive=100,
        )

        # Make a corner plot.
        fig = result.plot_corner()

        return result, fig


class HellingerDistance:

    def __init__(self, model_dir, N, waveform_generator):

        self._outdir = model_dir + "hellinger_cache/"

        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)

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

    def calculate_hellinger_distances(self, save=False, plot=False):

        if self.wg.extrinsic_at_train:
            print('dataset shouldn\'t be extrinsic at train')
            return

        rng = default_rng()
        idxs = rng.choice(self.wg.dataset_len, size=self.N, replace=False)

        for i in range(0, self.N):

            params_true = self.wg.params[idxs[i]]
            print('True Params:')
            print(params_true)

            cov_matrix = self.fisher.compute_analytical_cov_m1_m2_from_mu_chirp(params=params_true)
            print('Covariance matrix:')
            print(cov_matrix)
            #cov_matrix = self.fisher.compute_analytical_cov_m1_m2(params=params_true)
            #print('Covariance matrix:')
            #print(cov_matrix)

            bilby_result, _ = self.bilby.find_result(idxs[i], params_true)

            kde = bilby_result.kde
            #samples = np.transpose(bilby_result.samples) # shape: (# dims, # data)

            #print('# of samples')
            #print(len(bilby_result.samples))

            opt = optimize.minimize(lambda x: -kde(x), x0=params_true)

            print('Max likelihood point')
            print(opt.x)

            max_likelihood_pt = opt.x

            limit_low = max_likelihood_pt * 0.85
            limit_high = max_likelihood_pt * 1.15
            range_zoomed = np.stack((limit_low, limit_high), axis=1)

            # do binning on bilby samples
            # bilby_histogram, edges = np.histogramdd(bilby_result.samples, bins=8, density=True)
            #
            # for e in range(0, len(edges)):
            #     edges[e] = [(a + b) / 2 for a, b in zip(edges[e][:-1], edges[e][1:])]
            #
            # fisher_histogram = scipy.stats.multivariate_normal.pdf(np.array(np.meshgrid(edges[0], edges[1],
            #                                                     edges[2])).T.reshape(-1,3),
            #                                                      mean=max_likelihood_pt,
            #                                                      cov=cov_matrix)
            #print('shapes')
            #print(bilby_histogram.shape)
            #print(fisher_histogram.shape)

            #print('Edges')
            #print(edges)

            #bin_volume = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0])*(edges[2][1]-edges[2][0])
            #print('Bin volume: ')
            #print(bin_volume)

            #self.distances[i] = np.sqrt(np.sum(1./2 * ((np.sqrt(fisher_histogram) -
            #                                            np.sqrt(bilby_histogram.flatten()))**2)))
            #self.distances[i] = np.sqrt(self.distances[i])
            #print('Distance without bin volume')
            #print(self.distances[i])
            #self.distances[i] *= np.sqrt(bin_volume)
            #print('Distance with bin volume')
            #print(self.distances[i])

            # generate random samples
            sample_len = 10000
            mass_range = 5.
            dist_range = 100.

            ranges = np.asarray([mass_range, mass_range, dist_range])

            bin_volume = np.prod(ranges)/sample_len
            print('Bin volume')
            print(bin_volume)

            samples = np.random.random(size=(sample_len, 3))
            samples *= np.expand_dims(ranges, axis=0)
            samples += np.expand_dims(max_likelihood_pt, axis=0)

            print(samples[0, :])

            fisher_histogram = scipy.stats.multivariate_normal.pdf(samples,
                                                                mean=max_likelihood_pt,
                                                                cov=cov_matrix)
            #bilby_probabilities = np.transpose(kde.pdf(np.transpose(samples)))
            bilby_probabilities = np.zeros(sample_len)
            for b in range(0, sample_len):
                sample = dict(mass_1=samples[b,0], mass_2=samples[b,1],
                              luminosity_distance=samples[b,2])

                bilby_probabilities[b] = bilby_result.posterior_probability(sample)

            self.distances[i] = np.sqrt(np.sum(1./2 * ((np.sqrt(fisher_histogram) -
                                                 np.sqrt(bilby_probabilities))**2) * bin_volume))

            print('Distance with bin volume')
            print(self.distances[i])


            # for s in range(0, samples.shape[0]):
            #     self.distances[i] += 1./2*((np.sqrt(fisher_density[s]) -
            #                                       np.sqrt(kde.pdf(samples[s])))**2)
            #     dx += kde.pdf(samples[s])

            # dx = np.sum(kde.pdf(samples))

            # DO BINNING

            print('Iteration ' + str(i) + ' is complete')

            if plot:
                labels = ['m1', 'm2', 'dL']

                fig = corner.corner(bilby_result.samples, labels=labels, hist_kwargs={"density": True},
                                    bins=20, plot_datapoints=False, range=range_zoomed,
                                    no_fill_contours=False, fill_contours=True,
                                    levels=(0.3935, 0.8647, 0.9889, 0.9997),
                                    color='m', truths=params_true)

                plot_fisher_estimates(fig, range_zoomed, max_likelihood_pt, cov_matrix)

                if save:
                    fig.savefig(self._outdir + str(i))

                #plt.show()


            if save:
                np.savetxt(self._outdir+'true_params.txt', self.wg.params[idxs], delimiter=',')
                np.savetxt(self._outdir+'distances.txt', self.distances, delimiter=',')

        return self.distances

    def plot_hellinger_histogram(self):

        self.distances = np.loadtxt(self._outdir + 'distances.txt')
        plt.figure()
        plt.hist(self.distances, bins=25)
        plt.savefig(self._outdir + 'hellinger_histogram')


def plot_gauss_contours(params_true, cov_matrix, ind1, ind2, ax):

    # Initializing the covariance matrix
    cov = np.asarray([[cov_matrix[ind1, ind1], cov_matrix[ind1, ind2]],
                      [cov_matrix[ind1, ind2], cov_matrix[ind2, ind2]]])

    means = np.asarray([[params_true[ind1]], [params_true[ind2]]])

    w, v = np.linalg.eig(cov)

    t = np.linspace(0, 2*np.pi, num=100)
    xs = np.zeros((2, 100))

    # draw 1 sigma - 4 sigma
    for r in range(1, 5):

        xs[0,:] = r*np.sqrt(w[0])*np.cos(t)
        xs[1,:] = r*np.sqrt(w[1])*np.sin(t)

        xs_transformed = np.dot(v, xs)

        ax.plot(xs_transformed[0]+means[0], xs_transformed[1]+means[1], 'r')


def plot_fisher_estimates(fig, ranges, params_samples_ml, cov_matrix):

    "WORKS ONLY FOR 3 PARAMETERS"

    axes = fig.get_axes()

    # fisher 1d histograms
    for k in range(0, 3):

        x = np.linspace(ranges[k, 0], ranges[k, 1], 500)

        axes[4 * k].plot(x, scipy.stats.norm.pdf(x, loc=params_samples_ml[k],
                                    scale=np.sqrt(cov_matrix[k, k])), 'r-')

        for l in range(k + 1, 3):
            plot_gauss_contours(params_samples_ml, cov_matrix, k, l,
                                    axes[3 * l + k])

