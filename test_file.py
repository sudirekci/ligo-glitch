import sys

import torch

import bilby_posterior
import waveform_dataset as waveform_dataset
import waveform_dataset_3p as waveform_dataset_3p
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from pycbc.fft import backend_support
from fisher_info import Fisher
import corner


from bilby_posterior import Bilby_Posterior
from bilby_posterior import HellingerDistance

backend_support.set_backend(['mkl'])
np.set_printoptions(threshold=sys.maxsize)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

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

    Vh = dataset.svd.Vh

    print(np.dot(Vh, np.transpose(Vh.conj())))

    print(dataset.detector_signals)


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

def test_SVD_reconstruction():

    svd_no_basis_coeffs = 10

    dataset2 = waveform_dataset_3p.WaveformGenerator(dataset_len=10000, path_to_glitschen=path_to_glitschen,
                                                  extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                  add_glitch=False, svd_no_basis_coeffs=svd_no_basis_coeffs,
                                                     detectors=[0,1],sampling_frequency=512.,add_noise=False)

    dataset2.construct_signal_dataset(perform_svd=False)

    det_signals = dataset2.detector_signals

    dataset2.perform_svd()
    #dataset2.add_noise_to_detector_signals_after_SVD()

    det_signals_svd = dataset2.detector_signals

    print(det_signals.shape)
    print(det_signals_svd.shape)
    print(dataset2.svd.Vh.shape)

    plt.figure()
    for i in range(0, svd_no_basis_coeffs):
        plt.plot(np.abs(dataset2.svd.Vh[i]),alpha=0.4)

    #det_signals += np.fft.rfft(np.random.normal(0, scale=1.0,
    #                size=(dataset2.no_detectors,dataset2.dataset_len,dataset2.length)))[
    #                            :,:,dataset2.fft_mask] * dataset2.dt * \
    #                        np.sqrt(dataset2.bandwidth)

    for j in range(0, dataset2.no_detectors):
        for i in range(0, dataset2.dataset_len):

            reconstructed_sig = dataset2.svd.fseries(det_signals_svd[j,i,:])

            plt.figure()
            plt.plot(dataset2.freqs[dataset2.fft_mask],np.abs(reconstructed_sig),'b-')
            plt.plot(dataset2.freqs[dataset2.fft_mask],np.abs(det_signals[j,i,:]),'r-')
            plt.xlabel('Frequency (Hz)')
            p = dataset2.params[i]
            print(p)
            plt.title('Parameters: m1=%.1f, m2=%.2f, d=%.1f' % (p[0],p[1],p[2]))
            plt.show()


            #reconstructed_sig = np.fft.irfft(np.pad(reconstructed_sig, (1, 0), 'constant')) * \
            #                    dataset2.df * dataset2.length / np.sqrt(dataset2.bandwidth)

            #plt.figure()
            #plt.plot(reconstructed_sig, 'b-',alpha=0.4)
            #plt.plot(np.fft.irfft(np.pad(det_signals[j,i,:], (1, 0), 'constant')) * \
            #                   dataset2.df * dataset2.length / np.sqrt(dataset2.bandwidth),'r-',alpha=0.4)
            #plt.show()
            print(np.sum(np.abs(reconstructed_sig-det_signals[j,i,:])**2))


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


def test_fisher_step_size(el, ind1, ind2):

    f = Fisher(waveform_generator=dataset)

    cmap = plt.get_cmap('plasma')

    for i in range(0, dataset_len):

        start = 3
        end = 6

        ss = np.linspace(start, end, num=100)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_xscale('log')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_xscale('log')

        for j in range(0, len(ss)):

            step_size = 1./np.power(10, ss[j])

            f.step_sizes[el] = step_size
            #f.take_derivative(el)
            #color = cmap((ss[j] - start) / (end - start))

            f.compute_fisher_matrix(index=i)

            #plt.plot(np.abs(f.derivatives[el][0, :]), color=color)

            ax1.scatter(step_size, f.F[ind1, ind1], color='red')
            ax1.set_title(str(ind1)+' '+str(ind1))
            #plt.scatter(ss[j], np.imag(f.derivatives[el][1, 200]), color='blue')


        for j in range(0, len(ss)):

            step_size = 1. / np.power(10, ss[j])

            f.step_sizes[el] = step_size

            f.compute_fisher_matrix(index=i)

            ax2.scatter(step_size, f.F[ind1, ind2], color='blue')
            ax2.set_title(str(ind1) + ' ' + str(ind2))

        plt.show()


def test_fisher_cov():

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=True, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

    dataset.construct_signal_dataset(perform_svd=True, save=True, filename='test')

    f = Fisher(waveform_generator=dataset)

    for i in range(0, dataset_len):

        cov = f.compute_fisher_cov(index=i)

        print(cov)
        print('Positive def: ' , is_pos_def(cov))


def test_fisher_rms():

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=True, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

    dataset.construct_signal_dataset(perform_svd=True, save=True, filename='test')

    f = Fisher(waveform_generator=dataset)

    for i in range(0, dataset_len):

        _, th_cov = f.compute_theoretical_cov_mu_chirp(index=i)

        _, analy_cov = f.compute_analytical_cov_mu_chirp(index=i)

        print('*****************')
        print('Theoretical Cov Matrix Mu Chirp')
        print(th_cov)
        print('Analytical Cov Matrix Mu Chirp')
        print(analy_cov)
        print('*****************')

        analy_cov_1 = f.compute_analytical_cov_m1_m2(index=i)
        analy_cov_2 = f.compute_analytical_cov_m1_m2_from_mu_chirp(index=i)

        print('*****************')
        print('Analytical Cov Matrix M1 M2 1')
        print(analy_cov_1)
        print('Analytical Cov Matrix M1 M2 2')
        print(analy_cov_2)
        print('*****************')


def test_extrinsic_at_train():

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

    dataset.construct_signal_dataset(perform_svd=False, save=True, filename='test')

    print(dataset.hp.shape)
    #print(dataset.hp)
    print(dataset.hc.shape)
    #print(dataset.hc)
    print(dataset.means[0].shape)
    #print(dataset.stds[0].shape)

    #bb = ((dataset.hc-np.expand_dims(dataset.means[2], axis=0))/np.expand_dims(dataset.stds[2], axis=0))

    #print(np.mean(bb.real, axis=0))
    #print(np.std(bb.real, axis=0))

    dataset1 = waveform_dataset_3p.WaveformGenerator()
    dataset1.load_data(filename='test')

    print(dataset1.performed_svd)

    print(dataset1.hp.shape)
    #print(dataset.hp)
    print(dataset1.hc.shape)
    #print(dataset.hc)
    print(dataset1.means[0].shape)
    #print(dataset.stds[0].shape)

    #dataset1.initialize_svd()
    dataset1.perform_svd()
    dataset1.calculate_dataset_statistics()

    dataset1.normalize_params()
    dataset1.normalize_dataset()

    print('*********************')
    print(np.mean(dataset1.hc, axis=0))
    print(np.mean(dataset1.hp, axis=0))
    print(np.std(dataset1.hc, axis=0))
    print(np.std(dataset1.hp, axis=0))
    print('*********************')

    params_all = np.zeros(dataset_len*10)
    wfs_all = np.zeros((dataset_len*10, 4*svd_no_basis_coeffs))

    for i in range(0, dataset_len*10):
        wf, params = dataset1.provide_sample(i)
        #print(wf)
        params_all[i] = params[2]
        wfs_all[i] = wf
        print(wf)
        print(params)
        plt.figure()
        plt.plot(np.abs(dataset1.svd.fseries(wf[0:svd_no_basis_coeffs]+
                                             1j*wf[svd_no_basis_coeffs:2*svd_no_basis_coeffs])))

        plt.plot(np.abs(dataset1.svd.fseries(wf[2*svd_no_basis_coeffs:3*svd_no_basis_coeffs] +
                                             1j * wf[3*svd_no_basis_coeffs:4* svd_no_basis_coeffs])))
        plt.show()


def SVD_noise_test():

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs, duration=8.,
                                                    sampling_frequency=2048.)

    dataset.construct_signal_dataset(perform_svd=True, save=True, filename='test')

    print(len(dataset.freqs))
    print(dataset.bandwidth)
    print(dataset.dt)
    print(dataset.dt ** 2 * dataset.bandwidth * dataset.length)

    noises = np.zeros((1000, svd_no_basis_coeffs), dtype=complex)

    for j in range(0, 1000):
        noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(dataset.length)))[
                    dataset.fft_mask] * dataset.dt * \
                np.sqrt(dataset.bandwidth)
        noises[j, :] = dataset.svd.basis_coeffs(noise)

    print(np.mean(noises, axis=0))
    print(np.std(noises, axis=0))


def SVD_noise_test2():

    dataset_len = 10000
    dataset_len2 = 10
    svd_no_basis_coeffs = 10

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs, duration=8.,
                                                    sampling_frequency=2048.)

    dataset.construct_signal_dataset(perform_svd=True)

    print('Vh dagger Vh')
    for i in range(0,svd_no_basis_coeffs):
        print(np.sum(np.conjugate(dataset.svd.Vh[i])*dataset.svd.Vh[i]))

    dataset1 =  waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len2, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs, duration=8.,
                                                    sampling_frequency=2048.)

    dataset1.construct_signal_dataset(perform_svd=False)

    index1 = np.random.randint(low=0, high=dataset_len2)
    index2 = np.random.randint(low=0, high=dataset_len2)

    hp_hc_1 = np.random.randint(0, 2)
    hp_hc_2 = np.random.randint(0, 2)

    if hp_hc_1 == 0:
        h1 = dataset1.hp[index1]
    else:
        h1 = dataset1.hc[index1]

    if hp_hc_2 == 0:
        h2 = dataset1.hp[index2]
    else:
        h2 = dataset1.hc[index2]

    print('Normalize h1, h2 to have unit-SNR')

    h1 /= np.sqrt(dataset1.inner_whitened(h1, h1))
    h2 /= np.sqrt(dataset1.inner_whitened(h2, h2))

    print('<h1|h2> before SVD:')
    print(dataset1.inner_whitened(h1, h2))

    dataset1.perform_svd(Vh=dataset.svd.Vh)

    if hp_hc_1 == 0:
        h1_svd = dataset1.hp[index1]
    else:
        h1_svd = dataset1.hc[index1]

    if hp_hc_2 == 0:
        h2_svd = dataset1.hp[index2]
    else:
        h2_svd = dataset1.hc[index2]

    print('<h1|h2> after SVD:')
    print(np.real(np.sum(np.conjugate(h1_svd)*h2_svd))*4*dataset.df)

    noise_len = 1000

    n1h1 = np.zeros(noise_len)
    n1h1_svd = np.zeros(noise_len)
    n2h2 = np.zeros(noise_len)
    n2h2_svd = np.zeros(noise_len)

    for i in range(0, 1000):

        noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(dataset1.length)))[
                                dataset1.fft_mask] * dataset1.dt * \
                            np.sqrt(dataset1.bandwidth)

        n1h1[i] = dataset1.inner_whitened(noise, h1)

        n1h1_svd[i] = dataset1.inner_whitened(dataset.svd.basis_coeffs(noise), h1_svd)

        n2h2[i] = dataset1.inner_whitened(noise, h2)

        n2h2_svd[i] = dataset1.inner_whitened(dataset.svd.basis_coeffs(noise), h2_svd)

    print('Mean and std of <n|h1>:')
    print(np.mean(n1h1), np.std(n1h1))

    print('Mean and std of <n|h2>:')
    print(np.mean(n2h2), np.std(n2h2))

    print('Mean and std of nu.alpha1:')
    print(np.mean(n1h1_svd), np.std(n1h1_svd))

    print('Mean and std of nu.alpha2:')
    print(np.mean(n2h2_svd), np.std(n2h2_svd))

    print('Covariance of <n|h1> and <n|h2>:')
    print(np.mean((n1h1-np.mean(n1h1))*(n2h2-np.mean(n2h2))))

    print('<h1|h2> after SVD:')
    print(np.real(np.sum(np.conjugate(h1_svd) * h2_svd)) * 4 * dataset.df)


def test_compression():

    dataset_len = 10000

    basis_coeff_arr = [10, 20, 40, 60, 80, 100, 120, 160, 200, 240]
    #basis_coeff_arr = [10]
    differences = np.zeros(len(basis_coeff_arr))
    match = np.zeros(len(basis_coeff_arr))

    dataset_len2 = 100

    for i, svd_no_basis_coeffs in enumerate(basis_coeff_arr):

        dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

        dataset.construct_signal_dataset(perform_svd=True)

        dataset1 = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len2, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

        dataset1.construct_signal_dataset(save='False', perform_svd=False)

        original_hp = dataset1.hp/np.sqrt(dataset1.inner_whitened(dataset1.hp, dataset1.hp))
        original_hc = dataset1.hc/np.sqrt(dataset1.inner_whitened(dataset1.hc, dataset1.hc))

        reconstructed_hp = dataset.svd.fseries(dataset.svd.basis_coeffs(original_hp))
        reconstructed_hc = dataset.svd.fseries(dataset.svd.basis_coeffs(original_hc))

        # differences[i] = np.sqrt(np.sum(np.abs(np.nan_to_num((reconstructed_hp-original_hp)/
        #                                        original_hp))**2) +
        #                   np.sum(np.abs(np.nan_to_num((reconstructed_hc-original_hc)/
        #                                 original_hc))**2))/dataset_len2

        match[i] = np.abs(np.sum((np.conjugate(reconstructed_hc)*original_hc+np.conjugate(reconstructed_hp)*original_hp)))/2.

    print(match)

    plt.figure()
    plt.plot(basis_coeff_arr, 1-match)
    plt.yscale('log')
    plt.xlabel('SVD # of Basis Coefficients')
    plt.ylabel('1-match')
    plt.show()

def plot_signal_noise_glitch():

    dataset_len = 10

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

    dataset.construct_signal_dataset(perform_svd=False)

    dataset.initialize_glitch_matrices()

    td_axis = np.linspace(-dataset.duration / 2., dataset.duration / 2.,
                          num=int(dataset.sampling_freq * dataset.duration))

    #noises = np.zeros((dataset_len, dataset.no_detectors, int(dataset.length/2)),dtype=complex)
    #glitches = np.zeros((dataset_len, dataset.no_detectors, int(dataset.length/2)),dtype=complex)

    for i in range(0, dataset_len):

        m1 = dataset.params[i, dataset.INTRINSIC_PARAMS['mass1']]
        m2 = dataset.params[i, dataset.INTRINSIC_PARAMS['mass2']]
        distance = dataset.params[i, dataset.EXTRINSIC_PARAMS['distance']]

        dataset.projection_strains = []

        for j in range(0, dataset.no_detectors):
            dataset.projection_strains.append(np.zeros(int(dataset.length/2), dtype=complex))

        det, _ = dataset.add_glitch_to_projection_strains()

        glitch = np.copy(dataset.projection_strains[det])
        noise = np.fft.rfft(np.random.normal(0, scale=1.0, size=int(dataset.length)))[
                        dataset.fft_mask] * dataset.dt * \
                    np.sqrt(dataset.bandwidth)

        fig, ax = plt.subplots()
        line1, = ax.plot(dataset.freqs[dataset.fft_mask], np.abs(dataset.detector_signals[det, i, :]),
                         'b', alpha=0.5, label='signal')
        line2, = ax.plot(dataset.freqs[dataset.fft_mask], np.abs(glitch), 'r', alpha=0.5, label='glitch')
        line3, = ax.plot(dataset.freqs[dataset.fft_mask], np.abs(noise), 'g', alpha=0.5, label='noise')
        ax.legend(handles=[line1,line2,line3])
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Fourier Amplitude')
        plt.title('Parameters: m1=%.1f, m2=%.2f, d=%.1f' % (m1,m2,distance))
        plt.savefig('/home/su/Desktop/caltech/glitch_project/example_figs/' + str(i) + '_fourier')

        fig, ax = plt.subplots()
        line1, = ax.plot(td_axis, np.fft.irfft(np.pad(dataset.detector_signals[det, i, :], (1, 0),
                                              'constant')) * dataset.df * dataset.length /
                 np.sqrt(dataset.bandwidth), 'b', alpha=0.5, label='signal')
        line2, = ax.plot(td_axis, np.fft.irfft(np.pad(glitch, (1, 0),
                                              'constant')) * dataset.df * dataset.length /
                 np.sqrt(dataset.bandwidth), 'r', alpha=0.5, label='glitch')
        line3, = ax.plot(td_axis, np.fft.irfft(np.pad(noise, (1, 0),
                                              'constant')) * dataset.df * dataset.length /
                 np.sqrt(dataset.bandwidth), 'g', alpha=0.5, label='noise')
        ax.legend(handles=[line1, line2, line3])
        plt.xlabel('Time (s)')
        plt.ylabel('Whitened Waveforms')
        plt.title('Parameters: m1=%.1f, m2=%.2f, d=%.1f' % (m1, m2, distance))
        plt.savefig('/home/su/Desktop/caltech/glitch_project/example_figs/' + str(i))

def test_glitch_SVD_projection():

    dataset_len = 10000
    svd_no_basis_coeffs = 100

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs, duration=4.,
                                                    sampling_frequency=512.)

    dataset.construct_signal_dataset(perform_svd=True)

    average_pow = np.sum(np.sqrt(np.sum(np.abs(dataset.hp)**2, axis=1)) +
                          np.sqrt(np.sum(np.abs(dataset.hc)**2, axis=1)))/(2*dataset_len)
    print(average_pow)

    dataset.performed_svd = False
    dataset.initialize_glitch_matrices()

    glitch_len = 10000
    glitch_pow = np.zeros(glitch_len)

    for i in range(0, glitch_len):

        dataset.projection_strains = []

        for j in range(0, dataset.no_detectors):
            dataset.projection_strains.append(np.zeros(int(dataset.length / 2), dtype=complex))

        det, _ = dataset.add_glitch_to_projection_strains()

        glitch = np.copy(dataset.projection_strains[det])

        glitch_pow[i] = np.sqrt(np.sum(np.abs(dataset.svd.basis_coeffs(glitch))**2))
        print(glitch_pow[i])

    print(np.max(glitch_pow))

    # plt.figure()
    # plt.hist(glitch_pow, density=True, bins=100)
    # plt.xlabel('Relative amplitude after projection')
    # plt.ylabel('Density')
    # plt.show()


def test_glitch_SVD_basis():

    dataset_len = 10000
    N = 5000

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=True, add_noise=False, directory=directory)
    dataset.initialize()

    basis_coeff_arr = [50, 100, 150, 200, 250, 300]
    rel_errors = np.zeros((len(basis_coeff_arr), N))

    for ind, coeff in enumerate(basis_coeff_arr):

        Vh = dataset.create_glitch_SVD_basis(no_basis_coeffs=coeff)

        for i in range(0, N):

            glitch = dataset.create_FD_padded_glitch()

            # project and re-construct glitch
            glitch_reconstructed = (glitch @ Vh.T.conj()) @ Vh

            # plt.figure()
            # plt.plot(np.fft.irfft(glitch), '-r')
            # plt.plot(np.fft.irfft(glitch_reconstructed), '-b')
            # plt.show()

            rel_error = np.sqrt(np.sum(np.abs(glitch-glitch_reconstructed)**2)/np.sum(np.abs(glitch)**2))
            rel_errors[ind, i] = rel_error

        plt.figure()
        plt.hist(rel_errors[ind,:], bins=20, density=True)
        plt.title('Error for # basis coeffs = ' + str(coeff))
        plt.xlabel('Relative error')
        plt.ylabel('Density')
        plt.draw()

    plt.figure()
    plt.plot(basis_coeff_arr, np.sum(rel_errors, axis=1)/N)
    plt.title('Global Error Trend for Glitch SVD Projection')
    plt.xlabel('# of basis coeffs')
    plt.ylabel('Average relative error')

    plt.show()


def test_bilby():

    dataset_len = 5000

    #dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
    #                                                extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
    #                                                add_glitch=False, add_noise=False, directory=directory)

    #dataset.construct_signal_dataset(perform_svd=True, save=True, filename='bilby_test_dataset')
    #print('dataset constructed')

    #del dataset

    dataset1 = waveform_dataset_3p.WaveformGenerator(directory=directory)
    dataset1.load_data(filename='bilby_test_dataset')
    dataset1.normalize_params()

    bilbly_post = Bilby_Posterior(dataset1, model_dir='/home/su/Desktop/caltech/glitch_project/bilby_test/')

    idx = np.random.randint(0, dataset_len)

    y, params_true = dataset1.provide_sample(idx, return_det=False)
    params_true = dataset1.post_process_parameters(params_true)

    print('Params True')
    print(params_true)

    bilby_fig = bilbly_post.find_result(idx, params_true)


def test_hellinger(plot=False):

    dataset_len = 500

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                   extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                   add_glitch=False, add_noise=False, directory=directory)

    dataset.construct_signal_dataset(perform_svd=False)

    hellinger = bilby_posterior.HellingerDistance(model_dir=model_dir,
                                                  N=20, waveform_generator=dataset)
    print('hellinger initialized')

    hellinger.calculate_hellinger_distances(save=True, plot=plot)
    print("**************** MEAN DISTANCE **********************")
    print(hellinger.distance_mean())

def hellinger_histogram():

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=100, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory)
    dataset.initialize()

    hellinger = bilby_posterior.HellingerDistance(model_dir=model_dir, N=0, waveform_generator=dataset)
    hellinger.plot_hellinger_histogram()


def maximize_match_compute_posterior(n):

    dataset_len = n

    dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=False, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    sampling_frequency=2048., duration=8.)

    dataset.construct_signal_dataset(save=False, perform_svd=False)

    zs, glitches = dataset.create_glitch(length=n)

    dir_to_save = '/home/su/Desktop/caltech/glitch_project/matching_with_glitches/'

    # do sliding
    print('Dataset dt')
    print(dataset.dt)
    print('Dataset df')
    print(dataset.df)
    print(np.sqrt(dataset.bandwidth))
    print(np.sqrt(dataset.bandwidth)*dataset.dt)
    print(dataset.df)

    signal_length = dataset.length
    glitch_length = len(glitches[0])

    print('Length of a signal (in terms of array elements)')
    print(signal_length)
    print('Length of a glitch (in terms of array elements)')
    print(glitch_length)

    print('Max number of iterations')
    print(signal_length-glitch_length)

    no_iterations = min(signal_length-glitch_length, 200)

    # number of array elements that we skip per iteration
    dlength = int((signal_length - glitch_length) / no_iterations)

    matches = np.zeros((n, no_iterations))

    time_axis = np.arange(0, signal_length)/signal_length*dataset.duration-dataset.duration/2

    # choose a detector
    d = 0

    for i in range(0, n):

        # signal = dataset.detector_signals[d, i, :]

        signal = np.fft.irfft(np.pad(dataset.detector_signals[d, i, :]* \
                            np.exp(-1j*2*np.pi*dataset.freqs[dataset.fft_mask]*dataset.duration/2), (1, 0),
                            'constant')) * dataset.df * dataset.length /\
                            np.sqrt(dataset.bandwidth)
        snr = dataset.snrs[d, i]

        # find the norm of the signal
        norm_signal = np.sqrt(np.sum(signal*np.conjugate(signal)))
        print('norm of the signal')
        print(norm_signal)
        print('snr')
        print(snr)
        #plt.figure()
        #plt.plot(np.fft.irfft(signal)*dataset.length*dataset.df)
        #plt.plot(signal)
        #plt.show()

        # find the norm of the glitch
        norm_glitch = np.sqrt(np.sum(glitches[i, :]**2))
        # print('norm of the glitch')
        # print(norm_glitch)

        for j in range(0, no_iterations):

            matches[i, j] = np.abs(np.sum(signal[j*dlength:j*dlength+glitch_length]*glitches[i])\
                    / (norm_signal*norm_glitch))


    # find TOP_MAX best matches
    TOP_MAX = 5

    best_match_js = np.argmax(matches, axis=1)
    best_match_indices = np.argsort(matches[np.arange(0, n), best_match_js])[-TOP_MAX:]
    best_matches = matches[best_match_indices, np.argmax(matches, axis=1)[best_match_indices]]

    print(best_match_indices)
    print(best_matches)

    for i in range(0, TOP_MAX):

        true_params = dataset.params[best_match_indices[i]]
        snr = dataset.snrs[d, best_match_indices[i]]

        plt.figure()
        plt.plot(dataset.duration/2*(-1+(2*np.arange(0, no_iterations)*dlength+glitch_length)/signal_length),
                     matches[best_match_indices[i]])
        plt.title('Match')
        plt.xlabel('t (Relative to merge time)')
        plt.title('Best match = {match:.2f}'.format(match=best_matches[i]))
        plt.savefig(dir_to_save+str(best_match_indices[i])+'_match_plot.png')

        plt.figure()
        j = best_match_js[best_match_indices[i]]

        signal = np.fft.irfft(np.pad(dataset.detector_signals[d, best_match_indices[i], :]*\
                            np.exp(-1j*2*np.pi*dataset.freqs[dataset.fft_mask]*dataset.duration/2), (1, 0),
                            'constant')) * dataset.df * dataset.length /\
                            np.sqrt(dataset.bandwidth)

        plt.plot(time_axis, signal)
        plt.plot(time_axis[j*dlength:j*dlength+glitch_length],
                 glitches[best_match_indices[i]])
        plt.title('Parameters: m1=%.1f, m2=%.2f, d=%.1f snr=%.1f' % (true_params[0], true_params[1],
                                                                     true_params[2], snr))
        plt.savefig(dir_to_save+str(best_match_indices[i])+'_td_signal_glitch.png')

    # time for bilby

    bilbly_post = Bilby_Posterior(dataset, model_dir=dir_to_save)
    for i in range(0, TOP_MAX):

        # do the analysis without glitch

        true_params = dataset.params[best_match_indices[i]]
        print('true params')
        print(true_params)

        # bilby with glitch
        glitch_start_time = best_match_js[best_match_indices[i]]*dlength/\
                            dataset.length*dataset.duration-dataset.duration/2
        print(glitch_start_time)

        result2, _ = bilbly_post.find_result(idx=best_match_indices[i], params_true=true_params,
                                label=str(best_match_indices[i])+'_with_glitch',
                                glitch_array=glitches[best_match_indices[i]],
                                glitch_start_time=glitch_start_time)

        # bilby without glitch
        result1, _ = bilbly_post.find_result(idx=best_match_indices[i], params_true=true_params)

        fig = corner.corner(result1.samples, labels=['m1', 'm2', 'dL'], hist_kwargs={"density": True},
                      bins=20, plot_datapoints=False, no_fill_contours=False, fill_contours=True,
                      levels=(0.3935, 0.8647, 0.9889, 0.9997), color='b', truths=true_params)

        corner.corner(result2.samples, labels=['m1', 'm2', 'dL'], hist_kwargs={"density": True},
                      bins=20, plot_datapoints=False, no_fill_contours=False, fill_contours=True,
                      levels=(0.3935, 0.8647, 0.9889, 0.9997), color='r', fig=fig, truths=true_params)

        plt.savefig(dir_to_save + str(best_match_indices[i]) + '_bilby_results')



# dataset.initialize()

# test_noise()

# test_signals()

# test_waveform_dataset()

# test_saving_loading()

dataset_len = 10000
svd_no_basis_coeffs = 10

path_to_glitschen = '/home/su/Documents/glitschen-main/'
directory = '/home/su/Documents/glitch_dataset/'
#model_dir = '/home/su/Desktop/caltech/glitch_project/bilby_test/'

model_dir = '/home/su/Desktop/caltech/glitch_project/hellinger/'


# directory = '/home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_100_extrinsic_4/'
path_to_glitschen = '/home/su.direkci/programs/glitschen'
model_dir = '/home/su.direkci/glitch_project/hellinger_dist/'

#test_glitch_SVD_projection()

#SVD_noise_test2()

#test_compression()

#test_SVD()

#test_fisher_step_size('distance', 2, 0)

#print(dataset.snrs)


#test_fisher_rms()

#test_SVD_reconstruction()

#test_extrinsic_at_train()

#plot_signal_noise_glitch()


#dataset.normalize_params()
#print(dataset.performed_svd)

# dataset1 = waveform_dataset_3p.WaveformGenerator()
# dataset1.load_data(filename='test')
# dataset1.normalize_params()
#
# print(dataset1.extrinsic_at_train)
# print(dataset1.performed_svd)
#
# print(dataset1.params_mean)
# print(dataset1.params_std)
#
# f = Fisher(waveform_generator=dataset1)
#
# for i in range(0, dataset_len):
#
#     wf, params = dataset1.provide_sample(i)
#     print(params)
#
#     params = dataset1.post_process_parameters(params)
#
#     cov = f.compute_analytical_cov_m1_m2_from_mu_chirp(params=params)
#
#     print(cov)
#     print('Positive def: ' , is_pos_def(cov))

# test_glitch_SVD_basis()

# test_glitch_SVD_projection()

# test_bilby()

test_hellinger(plot=False)

#hellinger_histogram()

# hellinger_histogram()

#maximize_match_compute_posterior(100)
