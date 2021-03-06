import torch
from torch.utils.data import Dataset

import numpy as np
import glitschen
from glitschen import *
import pycbc.noise
import pycbc.psd
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.distributions import power_law

from numpy import fft
import sys
import h5py
import scipy
from scipy import signal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
import json
from sklearn.utils.extmath import randomized_svd



# f(x) = k*x^(n) dim = n+1
def power_law_rvs(n=2, length=10, dmin=100., dmax=1000.):
    dim = n + 1
    distrub = pycbc.distributions.power_law.UniformPowerLaw(dim=dim, d_L=[dmin, dmax])
    ds = distrub.rvs(size=length)

    return ds


class WaveformGenerator:

    INTRINSIC_PARAMS = dict(mass1=0, mass2=1, phase=2, a1=3, a2=4, theta1=5, theta2=6, phi_12=7, phi_JL=8,
                            theta_JN=9)
    INTRINSIC_LEN = len(INTRINSIC_PARAMS)
    EXTRINSIC_PARAMS = dict(distance=10, tc=11, right_ascension=12, declination=13, pol_angle=14)
    EXTRINSIC_LEN = len(EXTRINSIC_PARAMS)
    GLITCH_PARAMS = dict(time=15, z1=3, z2=4, z3=5, z4=6, z5=7)
    DETECTORS = dict(H1=0, L1=1)

    REF_TIME = 1e5
    SOLAR_MASS = 1.9891e+30

    def __init__(self, sampling_frequency=2048., duration=4., fmin=10., dataset_len=100000,
                 path_to_glitschen='/home/su/Documents/glitschen-main/',
                 q=5, winlen=0.5, approximant='IMRPhenomPv2', priors=None, detectors=None, tomte_to_blip=1,
                 extrinsic_at_train=False, directory='/home/su/Documents/glitch_dataset/', glitch_sigma=1, domain='FD',
                 svd_no_basis_coeffs=100, add_glitch=False, add_noise=False):

        self.stds = None
        self.means = None
        self.params_mean = None
        self.params_std = None
        self.glitch_params_mean = None
        self.glitch_params_std = None

        self.detector_list = detectors
        self.detectors = None
        self.no_detectors = None

        if priors is None:
            # default prior
            self.priors = np.zeros((self.INTRINSIC_LEN + self.EXTRINSIC_LEN + 1, 2))
            self.priors[self.INTRINSIC_PARAMS['mass1']] = [10., 80.]
            self.priors[self.INTRINSIC_PARAMS['mass2']] = [10., 80.]
            self.priors[self.EXTRINSIC_PARAMS['distance']] = [100., 1000.]
            self.priors[self.INTRINSIC_PARAMS['phase']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['a1']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['a2']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['theta1']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['theta2']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['phi_12']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['phi_JL']] = [0.0, 0.0]
            self.priors[self.INTRINSIC_PARAMS['theta_JN']] = [0.0, 0.0]
            self.priors[self.EXTRINSIC_PARAMS['tc']] = [0.0, 0.0]
            self.priors[self.EXTRINSIC_PARAMS['right_ascension']] = [0.0, 0.0]
            self.priors[self.EXTRINSIC_PARAMS['declination']] = [0.0, 0.0]
            self.priors[self.EXTRINSIC_PARAMS['pol_angle']] = [0.0, 0.0]
            self.priors[self.GLITCH_PARAMS['time']] = [-1.5, 1.5]

            #self.priors[self.INTRINSIC_PARAMS['phase']] = [0.0, 2 * np.pi]
            #self.priors[self.INTRINSIC_PARAMS['a1']] = [0., 0.88]
            #self.priors[self.INTRINSIC_PARAMS['a2']] = [0., 0.88]
            #self.priors[self.INTRINSIC_PARAMS['theta1']] = [0.0, np.pi]
            #self.priors[self.INTRINSIC_PARAMS['theta2']] = [0.0, np.pi]
            #self.priors[self.INTRINSIC_PARAMS['phi_12']] = [0.0, 2 * np.pi]
            #self.priors[self.INTRINSIC_PARAMS['phi_JL']] = [0.0, 2 * np.pi]
            #self.priors[self.INTRINSIC_PARAMS['theta_JN']] = [0.0, np.pi]
            #self.priors[self.EXTRINSIC_PARAMS['tc']] = [-0.1, 0.1]
            #self.priors[self.EXTRINSIC_PARAMS['right_ascension']] = [0.0, 2 * np.pi]
            #self.priors[self.EXTRINSIC_PARAMS['declination']] = [-np.pi / 2.0, np.pi / 2.0]
            #self.priors[self.EXTRINSIC_PARAMS['pol_angle']] = [0.0, np.pi]
            #self.priors[self.GLITCH_PARAMS['time']] = [-1.5, 1.5]
        else:
            self.priors = priors

        self.sampling_freq = sampling_frequency
        self.duration = duration

        self.length = None
        self.dt = None
        self.df = None
        self.freqs = None
        self.fft_mask = None

        self.add_noise = add_noise

        self.add_glitch = add_glitch
        self.path_to_glitschen = path_to_glitschen
        self.dataset_len = int(dataset_len)
        self.q = q
        self.winlen = winlen
        self.tomte_to_blip = tomte_to_blip
        self.fmin = fmin

        self.detector_signals = None
        self.params = None
        self.hp = None
        self.hc = None
        self.snrs = None
        self.extrinsic_at_train = extrinsic_at_train
        self.projection_strains = []
        self.domain = domain

        self.Z_blip = None
        self.W_blip = None
        self.Z_tomte = None
        self.W_tomte = None
        self.glitch_length = None
        self.glitch_params = None
        self.glitch_detector = None
        self.glitch_sigma = glitch_sigma

        self.bandwidth = None

        self.approximant = approximant
        self.directory = directory

        self.svd_no_basis_coeffs = svd_no_basis_coeffs
        self.svd = SVD(no_basis_coeffs=self.svd_no_basis_coeffs)
        self.performed_svd = False



    def initialize(self):

        self.initialize_freq_vars()
        self.initialize_detectors()
        self.calculate_psd()
        if self.add_glitch:
            self.initialize_glitch_matrices()
        self.calculate_params_statistics()
        self.calculate_dataset_statistics()
        # self.initialize_svd()



    def initialize_svd(self):

        self.svd = SVD(no_basis_coeffs=self.svd_no_basis_coeffs)
        self.performed_svd = False



    def initialize_freq_vars(self):

        self.length = int(self.duration * self.sampling_freq)
        self.dt = 1. / self.sampling_freq
        self.df = 1. / (self.dt * self.length)

        self.freqs = np.fft.rfftfreq(self.length) / self.dt
        self.fft_mask = self.freqs > 0



    def initialize_detectors(self):

        if self.detector_list is None:
            self.detector_list = [0, 1]

        self.no_detectors = len(self.detector_list)
        self.detectors = []

        key_list = list(self.DETECTORS.keys())
        val_list = list(self.DETECTORS.values())

        print('Initializing detectors:')

        for i, el in enumerate(self.detector_list):
            position = val_list.index(el)
            det_name = key_list[position]
            self.detectors.append(Detector(det_name))
            self.projection_strains.append(np.zeros(self.length))
            print(det_name)



    def calculate_psd(self):

        flen_noise = int(self.sampling_freq / 2 / self.df) + 2
        self.psd = pycbc.psd.aLIGOZeroDetHighPower(flen_noise, self.df, 0)[:-1]
        self.bandwidth = self.sampling_freq / 2



    def initialize_glitch_matrices(self):

        path1 = self.path_to_glitschen + '/examples/data/L1O3a_Tomtes_10-128.npy'
        path2 = self.path_to_glitschen + '/examples/data/L1O3a_Blips_10-512.npy'

        self.glitch_length = int(self.winlen * self.sampling_freq)

        self.W_tomte, _ = train_ppca(path1, q=self.q, srate=2048., winlen=self.winlen)
        # BLIPS HAVE DIFFERENT SAMPLING FREQ
        self.W_blip, _ = train_ppca(path2, q=self.q, srate=1024., winlen=self.winlen)

        # resample
        self.W_tomte = scipy.signal.resample(self.W_tomte, int(self.sampling_freq * self.winlen))
        self.W_blip = scipy.signal.resample(self.W_blip, int(self.sampling_freq * self.winlen))



    def create_glitch(self, name='tomte', length=10, mean=0):

        W, Z = None, None
        if name == 'tomte':
            W = self.W_tomte
        elif name == 'blip':
            W = self.W_blip

        Z_new = np.random.normal(loc=mean, scale=self.glitch_sigma, size=(self.q, length))

        gobs = np.matmul(W[:, 0:self.q], Z_new)
        mu = W[:, self.q]
        gobs += mu[:, np.newaxis]

        if length == 1:
            return np.squeeze(Z_new), np.squeeze(gobs)
        else:
            return Z_new, np.transpose(gobs)



    def generate_noise(self, length):

        return np.reshape(np.random.normal(0, 1, int(length * self.dataset_len)),
                          (self.dataset_len, length))



    def add_white_noise(self):

        return np.random.normal(0, 1, int(self.length))



    def inner_colored(self, freqseries1, freqseries2):

        inner = np.sum(freqseries1*np.conjugate(freqseries2)/self.psd[self.fft_mask])

        return np.real(inner)*4.*self.df



    def inner_whitened(self,freqseries1, freqseries2):

        psd = 1.0
        inner = np.sum(freqseries1*np.conjugate(freqseries2))/psd

        return np.real(inner)*4.*self.df



    def SNR_colored(self, freqseries1):

        return np.sqrt(self.inner_colored(freqseries1, freqseries1))



    def SNR_whitened(self, freqseries1):

        return np.sqrt(self.inner_whitened(freqseries1, freqseries1))



    def add_noise_to_projection_strains(self):

        for j in range(0, self.no_detectors):
            if self.domain == 'TD':
                noise = np.random.normal(0, scale=1.0, size=int(self.length))
            elif self.domain == 'FD':
                noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(self.length)))[self.fft_mask]*self.dt*\
                        np.sqrt(self.bandwidth)

            self.projection_strains[j] += noise



    def add_glitch_to_projection_strains(self):

        # sample glitch type
        if np.random.random() > self.tomte_to_blip:
            type = 0
            # sample blip
            z, glitch = self.create_glitch('blip', length=1)
        else:
            type = 1
            # sample tomte
            z, glitch = self.create_glitch('tomte', length=1)

        # sample time of glitch
        p = self.priors[self.GLITCH_PARAMS['time']]
        time = np.random.uniform(p[0], p[1])

        beginning = int((time - self.winlen / 2 + self.duration / 2) * self.sampling_freq)
        end = len(glitch) + beginning

        # sample detector
        det = np.random.randint(0, self.no_detectors)

        if self.domain == 'TD':

            self.projection_strains[det][beginning:end] += glitch

        elif self.domain == 'FD':

            glitch = np.fft.rfft(np.pad(glitch, (beginning, self.length-end), 'constant'))[self.fft_mask]*self.dt*\
                     np.sqrt(self.bandwidth)
            self.projection_strains[det] += glitch

        params = np.concatenate(([time], z))

        return det, params



    def get_spin_components(self, index):

        theta_jn = self.params[index][self.INTRINSIC_PARAMS['theta_JN']]
        mass1 = self.params[index][self.INTRINSIC_PARAMS['mass1']]*self.SOLAR_MASS
        mass2 = self.params[index][self.INTRINSIC_PARAMS['mass2']]*self.SOLAR_MASS
        phi_jl = self.params[index][self.INTRINSIC_PARAMS['phi_JL']]
        phi_12 = self.params[index][self.INTRINSIC_PARAMS['phi_12']]
        a1 = self.params[index][self.INTRINSIC_PARAMS['a1']]
        a2 = self.params[index][self.INTRINSIC_PARAMS['a2']]
        phase = self.params[index][self.INTRINSIC_PARAMS['phase']]
        tilt_1 = self.params[index][self.INTRINSIC_PARAMS['theta1']]
        tilt_2 = self.params[index][self.INTRINSIC_PARAMS['theta2']]

        if ((a1 == 0.0 or tilt_1 in [0, np.pi])
                and (a2 == 0.0 or tilt_2 in [0, np.pi])):
            spin1x = 0.0
            spin1y = 0.0
            spin1z = a1 * np.cos(tilt_1)
            spin2x = 0.0
            spin2y = 0.0
            spin2z = a2 * np.cos(tilt_2)
            iota = theta_jn
        else:
            iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                SimInspiralTransformPrecessingNewInitialConditions(theta_jn, phi_jl,
                                                                   tilt_1, tilt_2, phi_12, a1, a2,
                                                                   mass1, mass2,
                                                                   self.fmin, phase)

        return iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z



    def whiten_strain(self, ind, timeshift=0.):

        factor = 1.0

        if self.domain == 'TD':

            signal_fft = np.fft.rfft(self.projection_strains[ind])[self.fft_mask]

            snr = self.SNR_colored(signal_fft*self.dt)

            self.projection_strains[ind] = np.fft.irfft(np.pad(signal_fft * np.exp(-1j * 2 * np.pi *
                                                                self.freqs[self.fft_mask] * timeshift) *
                                                               (self.psd[self.fft_mask] * factor) ** (-0.5),
                                                               (1, 0), 'constant'))

        elif self.domain == 'FD':

            # divide by dt to match with TD
            # print('SNR COLORED: ', str(self.SNR_colored(self.projection_strains[ind])))

            snr = self.SNR_colored(self.projection_strains[ind])

            timeshift -= self.duration/2

            self.projection_strains[ind] = self.projection_strains[ind]* \
                    np.exp(-1j * 2 * np.pi * self.freqs[self.fft_mask] * timeshift) * \
                    (self.psd[self.fft_mask]) ** (-0.5)

        return snr



    def calculate_params_statistics(self):

        if self.params is not None:
            self.params_mean = np.mean(self.params, axis=0)
            self.params_std = np.std(self.params, axis=0)

        if self.add_glitch and self.glitch_params is not None:

            self.glitch_params_mean = np.mean(self.glitch_params, axis=0)
            self.glitch_params_std = np.std(self.glitch_params, axis=0)



    def calculate_dataset_statistics(self):

        if self.extrinsic_at_train:
            x = 0
            print('TODO')
        else:
            if self.detector_signals is not None:
                m1 = np.split(np.mean(self.detector_signals.real, axis=1), self.no_detectors)
                m2 = np.split(np.mean(self.detector_signals.imag, axis=1), self.no_detectors)
                self.means = m1+m2

                std1 = np.split(np.std(self.detector_signals.real, axis=1), self.no_detectors)
                std2 = np.split(np.std(self.detector_signals.imag, axis=1), self.no_detectors)
                self.stds = std1 + std2



    def compute_hp_hc(self, index):
        """
        Compute hps and hcs given the intrinsic parameters
        """

        # compute spinx, spiny, spinz
        iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = self.get_spin_components(index)

        if self.domain == 'TD':

            hp, hc = get_td_waveform(approximant=self.approximant,
                                     mass1=self.params[index][self.INTRINSIC_PARAMS['mass1']],
                                     mass2=self.params[index][self.INTRINSIC_PARAMS['mass2']],
                                     spin1x=spin1x,
                                     spin1y=spin1y,
                                     spin1z=spin1z,
                                     spin2x=spin2x,
                                     spin2y=spin2y,
                                     spin2z=spin2z,
                                     coa_phase=self.params[index][self.INTRINSIC_PARAMS['phase']],
                                     inclination=iota,
                                     delta_t=self.dt,
                                     f_lower=self.fmin)

            hp = hp.time_slice(-self.duration / 2, hp.sample_times[-1])
            hc = hp.time_slice(-self.duration / 2, hc.sample_times[-1])

            hp = np.pad(hp.data, (0, self.length - len(hp.data)), 'constant')
            hc = np.pad(hc.data, (0, self.length - len(hc.data)), 'constant')

        elif self.domain == 'FD':

            hp, hc = get_fd_waveform(approximant=self.approximant,
                                     mass1=self.params[index][self.INTRINSIC_PARAMS['mass1']],
                                     mass2=self.params[index][self.INTRINSIC_PARAMS['mass2']],
                                     spin1x=spin1x,
                                     spin1y=spin1y,
                                     spin1z=spin1z,
                                     spin2x=spin2x,
                                     spin2y=spin2y,
                                     spin2z=spin2z,
                                     coa_phase=self.params[index][self.INTRINSIC_PARAMS['phase']],
                                     inclination=iota,
                                     delta_f=self.df,
                                     f_lower=self.fmin,
                                     f_final=self.sampling_freq/2)

            hp = hp.data[self.fft_mask]
            hc = hc.data[self.fft_mask]

        return hp, hc



    def project_hp_hc(self, hp, hc, dataset_ind):

        # divide by the distance
        hp /= self.params[dataset_ind, self.EXTRINSIC_PARAMS['distance']]
        hc /= self.params[dataset_ind, self.EXTRINSIC_PARAMS['distance']]

        ra = self.params[dataset_ind, self.EXTRINSIC_PARAMS['right_ascension']]
        dec = self.params[dataset_ind, self.EXTRINSIC_PARAMS['declination']]
        polarization = self.params[dataset_ind, self.EXTRINSIC_PARAMS['pol_angle']]
        tc = self.params[dataset_ind, self.EXTRINSIC_PARAMS['tc']]

        snr_list = []

        for j in range(0, self.no_detectors):
            det = self.detectors[j]

            fp, fc = det.antenna_pattern(ra, dec, polarization, self.REF_TIME)

            dt = det.time_delay_from_earth_center(ra, dec, self.REF_TIME)
            self.projection_strains[j] = fp * hp + fc * hc

            snr = self.whiten_strain(j, timeshift=(dt + tc))
            snr_list.append(snr)

        # time shift and whiten
        # strain = self.whiten_strain(strain, timeshift=(dt+tc)*self.sampling_freq)

        # return snrs
        return snr_list



    def sample_intrinsic(self):
        """
        Sample intrinsic parameters
        """
        for i in range(0, self.INTRINSIC_LEN):
            p1 = self.priors[i]
            self.params[:, i] = np.random.uniform(p1[0], p1[1], self.dataset_len)

        # Ensure mass1 >= mass2
        mass1 = np.maximum(self.params[:, self.INTRINSIC_PARAMS['mass1']],
                           self.params[:, self.INTRINSIC_PARAMS['mass2']])
        mass2 = np.minimum(self.params[:, self.INTRINSIC_PARAMS['mass1']],
                           self.params[:, self.INTRINSIC_PARAMS['mass2']])

        self.params[:, self.INTRINSIC_PARAMS['mass1']] = mass1
        self.params[:, self.INTRINSIC_PARAMS['mass2']] = mass2

        del mass1
        del mass2



    def sample_extrinsic(self):

        # sample distance with power law
        distance_ind = self.EXTRINSIC_PARAMS['distance']
        self.params[:, distance_ind] = power_law_rvs(length=self.dataset_len, dmin=self.priors[distance_ind, 0],
                                                     dmax=self.priors[distance_ind, 1])

        # sample the rest uniformly
        for key, value in self.EXTRINSIC_PARAMS.items():
            if key != 'distance':
                self.params[:, value] = np.random.uniform(self.priors[value, 0],
                                                          self.priors[value, 1], self.dataset_len)



    def perform_svd(self, Vh=None):

        if Vh is not None:
            self.svd = SVD(self.svd_no_basis_coeffs, Vh)

        if self.extrinsic_at_train:

            if Vh is None:
                self.svd.generate_basis(np.vstack((self.hc, self.hp)))

            data1 = np.zeros((self.dataset_len, self.svd_no_basis_coeffs), dtype=np.complex_)
            data2 = np.zeros((self.dataset_len, self.svd_no_basis_coeffs), dtype=np.complex_)

            for i in range(0, self.dataset_len):
                data1[i] = self.svd.basis_coeffs(self.hp[i])
                data2[i] = self.svd.basis_coeffs(self.hc[i])

            self.hp = data1
            self.hc = data2

            del data1
            del data2

        else:

            if Vh is None:
                self.svd.generate_basis(np.reshape(self.detector_signals,
                                               (self.no_detectors * self.dataset_len, int(self.length / 2))))

            data = np.zeros((self.no_detectors, self.dataset_len, self.svd_no_basis_coeffs), dtype=np.complex_)

            for d in range(0, self.no_detectors):
                for i in range(0, self.dataset_len):

                    data[d, i, :] = self.svd.basis_coeffs(self.detector_signals[d, i, :])

            self.detector_signals = data
            del data

        self.performed_svd = True



    # this will be called to generate waveforms
    def construct_signal_dataset(self, save=False, filename = '', perform_svd=False):

        self.initialize()

        if self.domain == 'TD':
            length = self.length
            dtype = np.float
        elif self.domain == 'FD':
            length = int(self.length / 2)
            dtype = np.complex_

        if self.extrinsic_at_train:

            # compute hps and hcs, save them. Don't compute glitches or noise

            self.params = np.zeros((self.dataset_len, WaveformGenerator.INTRINSIC_LEN))
            self.sample_intrinsic()

            self.hp = np.zeros((self.dataset_len, length), dtype=dtype)
            self.hc = np.zeros((self.dataset_len, length), dtype=dtype)

            for i in range(0, self.dataset_len):
                self.hp[i], self.hc[i] = self.compute_hp_hc(i)

        else:
            # compute hps and hcs, sample extrinsic, project, add glitch and noise

            self.params = np.zeros((self.dataset_len, self.INTRINSIC_LEN + self.EXTRINSIC_LEN))
            self.detector_signals = np.zeros((self.no_detectors, self.dataset_len, length), dtype=dtype)
            if self.add_glitch:
                self.glitch_params = np.zeros((self.dataset_len, len(self.GLITCH_PARAMS)*self.no_detectors))
                self.glitch_detector = np.zeros(self.dataset_len)

            self.snrs = np.zeros((self.no_detectors, self.dataset_len))

            self.sample_intrinsic()
            self.sample_extrinsic()

            for i in range(0, self.dataset_len):
                hp, hc = self.compute_hp_hc(i)
                snrs = self.project_hp_hc(hp, hc, i)

                if self.add_noise:
                    self.add_noise_to_projection_strains()

                if self.add_glitch:
                    det, params = self.add_glitch_to_projection_strains()
                    self.glitch_params[i, det*len(self.GLITCH_PARAMS):(det+1)*len(self.GLITCH_PARAMS)] = params
                    self.glitch_detector[i] = int(det)

                for j in range(0, self.no_detectors):
                    self.detector_signals[j, i, :] = self.projection_strains[j]
                    self.snrs[j, i] = snrs[j]

        self.calculate_params_statistics()

        if perform_svd:

            self.perform_svd()
            print('SVD performed')

        self.calculate_dataset_statistics()

        if save:
            self.save_data(filename)



    def save_data(self, filename=''):

        f1 = h5py.File(self.directory + filename + '.hdf5', "w")

        config = {"sampling_freq": self.sampling_freq,
                  "duration": self.duration,
                  "fmin": self.fmin, "dataset_len": self.dataset_len,
                  "path_to_glitschen": self.path_to_glitschen,
                  "q": self.q, "winlen": self.winlen,
                  "approximant": self.approximant,
                  "tomte_to_blip": self.tomte_to_blip,
                  "extrinsic_at_train": self.extrinsic_at_train,
                  "glitch_sigma": self.glitch_sigma, "domain": self.domain,
                  "performed_svd": self.performed_svd,
                  "svd_no_basis_coeffs": self.svd_no_basis_coeffs,
                  "add_glitch": self.add_glitch,
                  "add_noise": self.add_noise}

        with open(self.directory + filename + '_config.json', 'w') as f:
            json.dump(config, f)

        f1.create_dataset("priors", self.priors.shape, dtype='f', data=self.priors)

        if self.performed_svd:
            f1.create_dataset("Vh_real", self.svd.Vh.shape, dtype='f', data=self.svd.Vh.real)
            f1.create_dataset("Vh_imag", self.svd.Vh.shape, dtype='f', data=self.svd.Vh.imag)

        if self.extrinsic_at_train:
            # save hp and hc
            # save intrinsic params
            if self.domain == 'TD':
                f1.create_dataset("hp", self.hp.shape, dtype='f', data=self.hp)
                f1.create_dataset("hc", self.hc.shape, dtype='f', data=self.hc)
            elif self.domain == 'FD':
                f1.create_dataset("hp_real", self.hp.shape, dtype='f', data=self.hp.real)
                f1.create_dataset("hp_imag", self.hp.shape, dtype='f', data=self.hp.imag)
                f1.create_dataset("hc_real", self.hc.shape, dtype='f', data=self.hc.real)
                f1.create_dataset("hc_imag", self.hc.shape, dtype='f', data=self.hc.imag)

            f1.create_dataset("intrinsic_params", self.params.shape,
                              dtype='f', data=self.params)

        else:
            # save signals
            # save intrinsic & extrinsic params, glitch params
            if self.domain == 'TD':
                f1.create_dataset("signals", self.detector_signals.shape,
                                dtype='f', data=self.detector_signals)
            elif self.domain == 'FD':
                f1.create_dataset("signals_real", self.detector_signals.shape,
                                dtype='f', data=self.detector_signals.real)
                f1.create_dataset("signals_imag", self.detector_signals.shape,
                                dtype='f', data=self.detector_signals.imag)
                f1.create_dataset("snrs", self.snrs.shape, dtype='f', data=self.snrs)

            f1.create_dataset("params", self.params.shape,
                              dtype='f', data=self.params)
            if self.add_glitch:
                f1.create_dataset("glitch_params", self.glitch_params.shape,
                              dtype='f', data=self.glitch_params)
                f1.create_dataset("glitch_detector", self.glitch_detector.shape,
                                  dtype='f', data=self.glitch_detector)

        f1.close()



    def provide_sample(self, idx):

        if self.extrinsic_at_train:

            x = 0
            print('TODO')
            return -1
            wf = None
            params = None

        else:

            params = np.nan_to_num((self.params[idx] - self.params_mean) / self.params_std)

            if self.add_glitch:
                glitch_params = np.nan_to_num((self.glitch_params[idx] - self.glitch_params_mean) /
                                              self.glitch_params_std)

                params = np.concatenate((params, glitch_params))

            # Concatenate the waveforms from the different detectors
            wfs = []
            for d in range(0, self.no_detectors):

                wf = self.detector_signals[d, idx, :]

                if self.domain == 'TD':

                    x = 0
                    print('TODO')

                elif self.domain == 'FD':

                    index = d * self.no_detectors

                    wf1 = (wf.real - np.squeeze(self.means[index])) / np.squeeze(self.stds[index])
                    wfs.append(wf1)
                    wf2 = (wf.imag - np.squeeze(self.means[index + 1])) / np.squeeze(self.stds[index + 1])
                    wfs.append(wf2)

            wf = np.concatenate(wfs, axis=-1)

        return wf, params



    def post_process_parameters(self, x):


        if not self.add_glitch:

            return x*self.params_std+self.params_mean

        # intrinsic & extrinsic params
        x1 = x[:,0:(self.INTRINSIC_LEN + self.EXTRINSIC_LEN)]
        x1 = x1*self.params_std + self.params_mean

        x = x[:,(self.INTRINSIC_LEN + self.EXTRINSIC_LEN):]
        x = x*self.glitch_params_std+self.glitch_params_mean

        return np.concatenate((x1,x), axis=-1)



    def load_data(self, filename):

        with open(self.directory + filename + '_config.json', "r") as jsonfile:
            obj_params = json.load(jsonfile)

        self.sampling_freq = obj_params['sampling_freq']
        self.duration = obj_params['duration']
        self.fmin = obj_params['fmin']
        self.dataset_len = obj_params['dataset_len']
        self.path_to_glitschen = obj_params['path_to_glitschen']
        self.q = obj_params['q']
        self.winlen = obj_params['winlen']
        self.approximant = obj_params['approximant']
        self.tomte_to_blip = obj_params['tomte_to_blip']
        self.extrinsic_at_train = obj_params['extrinsic_at_train']
        self.glitch_sigma = obj_params['glitch_sigma']
        self.domain = obj_params['domain']
        self.performed_svd = obj_params['performed_svd']
        self.svd_no_basis_coeffs = obj_params['svd_no_basis_coeffs']
        self.add_glitch = obj_params['add_glitch']
        self.add_noise = obj_params['add_noise']

        print('Add glitch is ', self.add_glitch)

        f2 = h5py.File(self.directory + filename + '.hdf5', 'r')

        self.priors = f2['priors'][:, :]

        if self.performed_svd:
            Vh = f2['Vh_real'][:]+1j*f2['Vh_imag'][:]
            self.svd = SVD(no_basis_coeffs=self.svd_no_basis_coeffs, Vh=Vh)

        if self.extrinsic_at_train:
            if self.domain == 'TD':
                self.hp = f2['hp'][:, :]
                self.hc = f2['hc'][:, :]
            elif self.domain == 'FD':
                self.hp = f2['hp_real'][:, :] + f2['hp_imag'][:, :]*1j
                self.hc = f2['hc_real'][:, :] + f2['hc_imag'][:, :]*1j

            self.params = f2['intrinsic_params'][:, :]  # this array will need to increase in size

        else:
            if self.domain == 'TD':
                self.detector_signals = f2['signals'][:, :, :]

            elif self.domain == 'FD':
                self.detector_signals = f2['signals_real'][:, :] + f2['signals_imag'][:, :]*1j
                self.snrs = f2['snrs']

            self.params = f2['params'][:, :]

            if self.add_glitch:
                self.glitch_params = f2['glitch_params'][:]
                self.glitch_detector = f2['glitch_detector'][:]

        self.initialize()

        f2.close()


class WaveformDatasetTorch(Dataset):

    """Wrapper for a WaveformDataset to use with PyTorch DataLoader."""

    def __init__(self, waveform_generator):

        self.wfg = waveform_generator

    def __len__(self):

        return self.wfg.dataset_len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        wf, params = self.wfg.provide_sample(idx)

        return torch.from_numpy(wf), torch.from_numpy(params)


class SVD:

    def __init__(self, no_basis_coeffs=100, Vh=None):

        self.no_basis_coeffs = no_basis_coeffs
        self.Vh = Vh
        if self.Vh is not None:
            self.V = self.Vh.T.conj()

    def generate_basis(self, data):

        U, s, Vh = randomized_svd(data, self.no_basis_coeffs)

        self.Vh = Vh.astype(np.complex64)
        self.V = self.Vh.T.conj()


    def fseries(self, coeffs):

        return coeffs @ self.Vh

    def basis_coeffs(self, fseries):

        return fseries @ self.V

