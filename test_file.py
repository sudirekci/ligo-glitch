import torch

import waveform_dataset as waveform_dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from pycbc.fft import backend_support

backend_support.set_backend(['mkl'])

dataset_len = 100
path_to_glitschen = '/home/su/Documents/glitschen-main/'

dataset = waveform_dataset.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                             extrinsic_at_train=False, tomte_to_blip=1, domain='FD', add_glitch=False,
                                             add_noise=False)


def inner_colored(freqseries1, freqseries2):
    inner = np.sum(freqseries1 * np.conjugate(freqseries2) / dataset.psd[dataset.fft_mask])

    return np.real(inner) * 4. * 0.25


def inner_whitened(freqseries1, freqseries2):
    psd = 1.0
    inner = np.sum(freqseries1 * np.conjugate(freqseries2)) / psd

    return np.real(inner) * 4. * 0.25


def plot_signals(dataset=None, index=0, domain='TD', loglog=False):

    dataset_domain = dataset.domain
    arr = dataset.detector_signals[:, index, :]

    fd_axis = dataset.freqs[1:]
    td_axis = np.linspace(-dataset.duration / 2., dataset.duration / 2.,
                          num=int(dataset.sampling_freq * dataset.duration))

    if dataset_domain == 'FD':

        if domain == 'FD':

            signals = np.abs(arr)

            if loglog:

                plt.figure()
                plt.loglog(fd_axis, signals[0])
                plt.loglog(fd_axis, signals[1])
                plt.figure()
                plt.semilogx(fd_axis, np.unwrap(np.angle(signals[0])))
                plt.semilogx(fd_axis, np.unwrap(np.angle(signals[1])))
                plt.show()

            else:
                plt.figure()
                plt.scatter(fd_axis, signals[0])
                plt.scatter(fd_axis, signals[1])
                plt.show()

        elif domain == 'TD':

            signals = np.fft.irfft(np.pad(arr, ((0, 0), (1, 0)), 'constant')) * dataset.df * dataset.length / np.sqrt(
                dataset.bandwidth)

            plt.figure()
            plt.plot(td_axis, signals[0])
            plt.plot(td_axis, signals[1])
            plt.show()

    elif dataset_domain == 'TD':

        if domain == 'TD':

            print(dataset.SNR_whitened(np.fft.rfft(arr[0] * dataset.dt)))
            print(dataset.SNR_whitened(np.fft.rfft(arr[1] * dataset.dt)))

            plt.figure()
            plt.plot(td_axis, arr[0])
            plt.plot(td_axis, arr[1])
            plt.show()

        elif domain == 'FD':

            print('TODO')


def test_noise():
    dataset_len = 20

    dataset = waveform_dataset.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                 extrinsic_at_train=False, tomte_to_blip=1, domain='FD')

    dataset.initialize()

    white_noise = np.fft.rfft(np.random.normal(0, 1, 8192) * 1. / 8192)

    print('White noise SNR: ', str(np.sqrt(inner_whitened(white_noise, white_noise))))

    normalized_noise = np.pad(white_noise[dataset.fft_mask] * dataset.dt * (dataset.psd[dataset.fft_mask])
                              ** (0.5), (1, 0), 'constant')
    normalized_noise /= np.sqrt(inner_colored(normalized_noise, normalized_noise))

    print('Normalized noise SNR: ', str(np.sqrt(inner_colored(normalized_noise, normalized_noise))))
    variates = []

    for k in range(0, 1000):
        white_noise = np.random.normal(0, 1, 8192)

        colored_noise = np.pad(np.fft.rfft(white_noise)[dataset.fft_mask] * dataset.dt * (dataset.psd[dataset.fft_mask]
        ) ** (0.5), (1, 0), 'constant')

        variates.append(inner_colored(colored_noise, normalized_noise))

    x = np.linspace(-3, 3, 1000)
    plt.figure()
    plt.hist(np.array(variates), bins=30, density=True)
    plt.plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2))
    plt.show()


def test_signals():
    dataset_domain = 'FD'
    dataset_len = 100

    dataset = waveform_dataset.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                 extrinsic_at_train=False, tomte_to_blip=1, domain=dataset_domain,
                                                 add_glitch=False, add_noise=False)

    dataset.construct_signal_dataset()

    params = dataset.params
    glitch_params = dataset.glitch_params

    def find_glitch_type(param):
        if int(param) == 0:
            return 'Blip'
        else:
            return 'Tomte'

    for i in range(0, dataset_len):
        print('mass 1, mass 2:')
        print(params[i, dataset.INTRINSIC_PARAMS['mass1']])
        print(params[i, dataset.INTRINSIC_PARAMS['mass2']])
        print('Distance:')
        print(params[i, dataset.EXTRINSIC_PARAMS['distance']])
        # print('Spin magnitudes:')
        # print(params[i][dataset.INTRINSIC_PARAMS['a1']])
        # print(params[i][dataset.INTRINSIC_PARAMS['a2']])
        # print('Right Ascension:')
        # print(params[i][dataset.EXTRINSIC_PARAMS['right_ascension']])
        # print('Declination:')
        # print(params[i][dataset.EXTRINSIC_PARAMS['declination']])
        # print('Polarization Angle:')
        # print(params[i][dataset.EXTRINSIC_PARAMS['pol_angle']])
        # print('tc GW signal:')
        # print(params[i][dataset.EXTRINSIC_PARAMS['tc']])
        print('Glitch params:')
        # print(glitch_params[i])
        print('SNRs:')
        print(dataset.snrs[0, i])
        print(dataset.snrs[1, i])
        # print(dataset.SNR_whitened(dataset.detector_signals[0,i,:]))

        plot_signals(dataset=dataset, index=i, domain='FD', loglog=False)

        # print('Glitch type:')
        # print(find_glitch_type(glitch_params[i][dataset.GLITCH_PARAMS['glitch_type']]))

        print('-----------------------------------------------')


def test_SVD():
    dataset_len = 20
    svd_no_basis_coeffs = 5

    dataset = waveform_dataset.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                 extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                 svd_no_basis_coeffs=svd_no_basis_coeffs, add_noise=True)

    dataset.construct_signal_dataset(perform_svd=True)
    Vh = dataset.svd.Vh

    # for i in range(0, svd_no_basis_coeffs):
    #     plt.figure()
    #     plt.plot(dataset.freqs[1:], np.abs(Vh[i]))
    #     plt.show()

    plt.figure()
    plt.title('Real parts of basis coeffs')
    for i in range(0, dataset_len):
        for j in range(0, dataset.no_detectors):
            plt.scatter(np.arange(start=1, stop=svd_no_basis_coeffs + 1),
                        np.real(dataset.detector_signals[j, i, :]))

    plt.figure()
    plt.title('Imaginary parts of basis coeffs')
    for i in range(0, dataset_len):
        for j in range(0, dataset.no_detectors):
            plt.scatter(np.arange(start=1, stop=svd_no_basis_coeffs + 1),
                        np.imag(dataset.detector_signals[j, i, :]))

    plt.show()

    for i in range(0, dataset_len):
        for j in range(0, dataset.no_detectors):
            plt.figure()
            plt.scatter(np.arange(start=1, stop=svd_no_basis_coeffs + 1),
                        np.real(dataset.detector_signals[j, i, :]))
            plt.scatter(np.arange(start=1, stop=svd_no_basis_coeffs + 1),
                        np.imag(dataset.detector_signals[j, i, :]))
            plt.show()


def test_waveform_dataset():
    dataset1 = waveform_dataset.WaveformGenerator(dataset_len=100, path_to_glitschen=path_to_glitschen,
                                                  extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                  add_glitch=True)

    dataset2 = waveform_dataset.WaveformGenerator(dataset_len=10, path_to_glitschen=path_to_glitschen,
                                                  extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                  add_glitch=True)

    dataset1.construct_signal_dataset(perform_svd=True)
    dataset2.construct_signal_dataset(perform_svd=False)

    dataset2.perform_svd(dataset1.svd.Vh)
    dataset2.calculate_dataset_statistics()
    # training dataset
    training_dataset = waveform_dataset.WaveformDatasetTorch(dataset1)
    print('training dataset prepared')

    testing_dataset = waveform_dataset.WaveformDatasetTorch(dataset2)

    print('testing dataset prepared')

    tr = []
    tr2 = []
    for i in range(0, 10):
        # print(testing_dataset.__getitem__(i))
        x, y = testing_dataset.__getitem__(i)
        tr.append(x.cpu().detach().numpy())
        tr2.append(y.cpu().detach().numpy())

        print(y)

    # tr = torch.cat(tr, -1).cpu().detach().numpy()
    tr = np.reshape(np.asarray(tr), (10, 400))
    tr2 = np.reshape(np.asarray(tr2), (10, 27))
    # print(np.std(tr, axis=0))
    # print(np.std(tr2, axis=0))


def test_saving_loading():
    dataset1 = waveform_dataset.WaveformGenerator(dataset_len=100, path_to_glitschen=path_to_glitschen,
                                                  extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                  add_glitch=True)

    dataset1.construct_signal_dataset(perform_svd=True)
    dataset1.save_data('test')

    dataset2 = waveform_dataset.WaveformGenerator()
    dataset2.load_data('test')

    print(dataset2.detector_signals)
    print(dataset2.params)
    print(dataset2.glitch_params)


# dataset.initialize()

# test_noise()

# test_signals()

# test_waveform_dataset()

# test_SVD()

# test_saving_loading()


dataset.construct_signal_dataset()
for i in range(0, dataset_len):
    plot_signals(dataset=dataset, index=i)
