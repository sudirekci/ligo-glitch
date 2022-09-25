import torch

import waveform_dataset as waveform_dataset
import waveform_dataset_3p as waveform_dataset_3p
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from pycbc.fft import backend_support
from fisher_info import Fisher

backend_support.set_backend(['mkl'])

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



# dataset.initialize()

# test_noise()

# test_signals()

# test_waveform_dataset()

# test_saving_loading()

dataset_len = 1000
svd_no_basis_coeffs = 4

path_to_glitschen = '/home/su/Documents/glitschen-main/'
directory='/home/su/Documents/glitch_dataset/'

#test_SVD()

#test_fisher_step_size('distance', 2, 0)

#print(dataset.snrs)


#test_fisher_rms()

#test_SVD_reconstruction()

#test_extrinsic_at_train()


dataset = waveform_dataset_3p.WaveformGenerator(dataset_len=dataset_len, path_to_glitschen=path_to_glitschen,
                                                    extrinsic_at_train=True, tomte_to_blip=1, domain='FD',
                                                    add_glitch=False, add_noise=False, directory=directory,
                                                    svd_no_basis_coeffs=svd_no_basis_coeffs)

dataset.initialize()
print(len(dataset.freqs))
print(dataset.bandwidth)
print(dataset.dt)

