import os
import argparse
from torch.utils.data import DataLoader
import torch
from pathlib import Path

import matplotlib.pyplot as plt
import corner
import csv
import time
import numpy as np
import h5py

import nde_flows
import waveform_dataset_3p as wd
from bilby_posterior import Bilby_Posterior

from fisher_info import Fisher
from scipy.stats import norm
import scipy.ndimage
from scipy.stats import multivariate_normal


#os.environ['OMP_NUM_THREADS'] = str(1)
#os.environ['MKL_NUM_THREADS'] = str(1)


"""
python gwpe.py train new nde \
    --data_dir /home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_100_extrinsic_4/ \
    --model_dir /home/su.direkci/glitch_project/models_no_glitch_w_noise/3d_31/ \
    --nbins 2 \
    --num_transform_blocks 1 \
    --nflows 3 \
    --batch_norm \
    --lr 0.0002 \
    --epochs 25 \
    --hidden_dims 32 \
    --activation elu \
    --no_lr_annealing \
    --batch_size 8000 \
    
    python gwpe.py train new nde \
    --data_dir /home/su.direkci/glitch_project/dataset_w_glitch_3p_svd_100_extrinsic/ \
    --model_dir /home/su.direkci/glitch_project/models/3d_1/ \
    --nbins 8 \
    --num_transform_blocks 1 \
    --nflows 15 \
    --batch_norm \
    --lr 0.0002 \
    --epochs 10 \
    --hidden_dims 32 \
    --activation elu \
    --lr_anneal_method cosine \
    --batch_size 2000 \

python gwpe.py train existing \
    --data_dir /home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_100_extrinsic_4/ \
    --model_dir /home/su.direkci/glitch_project/models_no_glitch_w_noise/3d_31/ \
    --epochs 30 \
    --batch_size 8000 \

python gwpe.py train existing \
    --data_dir /home/su.direkci/glitch_project/dataset_w_glitch_3p_svd_100_extrinsic/ \
    --model_dir /home/su.direkci/glitch_project/models/3d_1/ \
    --epochs 10 \
    --batch_size 2000 \


python gwpe.py test \
    --data_dir /home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_100_extrinsic_4/ \
    --model_dir /home/su.direkci/glitch_project/models_no_glitch_w_noise/3d_31/ \
    --bilby \
    --fisher \
    --test_on_training_data \
    
python gwpe.py test \
    --data_dir /home/su.direkci/glitch_project/dataset_w_glitch_3p_svd_100_extrinsic/ \
    --model_dir /home/su.direkci/glitch_project/models/3d_1/ \
    --fisher \
    --epoch 8\
    --test_on_training_data \
"""

#parameter_labels = np.asarray([r'$m_1$',r'$m_2$', r'$\phi_c$',r'$a_1$',r'$a_2$',r'$t_1$',r'$t_2$',
#                    r'$\phi_{12}$',r'$\phi_{jl}$',r'$\theta_{JN}$',r'$d_L$',r'$t_c$',
#                    r'$\alpha$',r'$\delta$',r'$\psi$', r'$t_{g1}$',r'$c_{11}$',
#                    r'$c_{12}$',r'$c_{13}$',r'$c_{14}$',r'$c_{15}$',r'$t_{g2}$',
#                    r'$c_{21}$',r'$c_{22}$',r'$c_{23}$',r'$c_{24}$',r'$c_{25}$'])

parameter_labels = np.asarray([r'$m_1$',r'$m_2$', r'$d_L$',r'$t_{c1}$',r'$c_{11}$',
                               r'$c_{12}$',r'$c_{13}$',r'$c_{14}$',r'$c_{15}$',
                               r'$t_{c2}$', r'$c_{21}$',r'$c_{22}$', r'$c_{23}$',
                               r'$c_{24}$', r'$c_{25}$'])

#extrinsic_factor = 100


class PosteriorModel(object):

    def __init__(self, model_dir=None, data_dir=None,
                 use_cuda=True):

        # self.wfd = None
        self.model = None
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_type = None
        self.optimizer = None
        self.scheduler = None
        self.detectors = None
        self.train_history = []
        self.test_history = []
        self.train_kl_history = []
        self.test_kl_history = []
        self.train_loader = None
        self.test_loader = None
        self.training_wg = None
        self.validation_wg = None
        self.testing_wg = None
        self.lr = None
        self.test_on_training_data = None
        self.epoch_to_use = None
        self.batch_size = None
        self.annealing = 'None'
        self.fisher = None
        self.bilbly_post = None

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')



    def load_dataset(self, batch_size=512):
        """
        Load database of waveforms and set up data loaders.
        Args:
            batch_size (int):  batch size for DataLoaders
        """

        # load training data
        self.training_wg = wd.WaveformGenerator(directory=self.data_dir)
        self.training_wg.load_data('training_data')
        print(self.training_wg.params[0:100])

        # validation data
        self.validation_wg = wd.WaveformGenerator(directory=self.data_dir)
        self.validation_wg.load_data('validation_data')

        if self.training_wg.extrinsic_at_train:
            # normalize the datasets in advance
            self.training_wg.normalize_params()
            self.validation_wg.normalize_params()

        wfd_train = wd.WaveformDatasetTorch(self.training_wg)
        wfd_test = wd.WaveformDatasetTorch(self.validation_wg)

        # DataLoader objects
        self.train_loader = DataLoader(
            wfd_train, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=16,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)),
            generator=torch.Generator(device='cuda'))

        self.test_loader = DataLoader(
            wfd_test, batch_size=batch_size, shuffle=False, pin_memory=True,
            num_workers=16,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))

        print('Datasets initialized')



    def construct_model(self, model_type, existing=False, **kwargs):
        """Construct the neural network model.
        Args:
            model_type:     'maf' or 'cvae'
            wfd:            (Optional) If constructing the model from a
                            WaveformDataset, include this. Otherwise, all
                            arguments are passed through kwargs.
            kwargs:         Depends on the model_type
                'maf'   input_dim       Do not include with wfd
                        context_dim     Do not include with wfd
                        hidden_dims
                        nflows
                        batch_norm      (True)
                        bn_momentum     (0.9)
                        activation      ('elu')
                'cvae'  input_dim       Do not include with wfd
                        context_dim     Do not include with wfd
                        latent_dim      int
                        hidden_dims     same for encoder and decoder
                                        list of ints
                        encoder_full_cov (True)
                        decoder_full_cov (True)
                        activation      ('elu')
                        batch_norm      (False)
                        iaf             Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        IAF. Keys should be:
                                            context_dim
                                            hidden_dims
                                            nflows
                        prior_maf     Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        MAF. Keys should be:
                                            hidden_dims
                                            nflows
                                        Note that this is conditioned on
                                        the waveforms automatically.
            * it is recommended to only use one of iaf or prior_maf
                        decoder_maf     Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        MAF. Keys should be:
                                            hidden_dims
                                            nflows
                                        Note that this is conditioned on
                                        the waveforms automatically.
        """

        if model_type == 'nde':
            model_creator = nde_flows.create_NDE_model
        else:
            raise NameError('Invalid model type')

        if not existing:
            # input_dim = self.wfd.nparams !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # context_dim = self.wfd.context_dim !!!!!!!!!!!!!!!!!!!!!!!!!!!
            context_dim = self.training_wg.svd_no_basis_coeffs*self.training_wg.no_detectors*2
            input_dim = self.training_wg.input_dim

            self.model = model_creator(input_dim=input_dim,
                                       context_dim=context_dim,
                                       **kwargs)
        else:
            self.model = model_creator(**kwargs)

        # Base distribution for sampling

        # I would like to use the code below, but the KL divergence doesn't
        # work... Should be a workaround.

        # self.base_dist = torch.distributions.Independent(
        #     torch.distributions.Normal(
        #         loc=torch.zeros(base_dim, device=self.device),
        #         scale=torch.ones(base_dim, device=self.device)
        #         ),
        #     1
        # )

        self.model.to(self.device)

        self.model_type = model_type



    def initialize_training(self, lr=0.0001,
                            lr_annealing=True, anneal_method='step',
                            total_epochs=None,
                            steplr_step_size=80, steplr_gamma=0.5,
                            flow_lr=None):
        """Set up the optimizer and scheduler."""

        self.lr = lr

        if self.model is None:
            raise NameError('Construct model before initializing training.')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        if lr_annealing is True:
            if anneal_method == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=steplr_step_size,
                    gamma=steplr_gamma)
            elif anneal_method == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=500,
                )
            elif anneal_method == 'cosineWR':
                self.scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2
                    )
                )

        self.epoch = 1


    def save_model(self, filename='model',
                   aux_filename='waveforms_supplementary.hdf5'):
        """Save a model and optimizer to file.
        Args:
            model:      model to be saved
            optimizer:  optimizer to be saved
            epoch:      current epoch number
            model_dir:  directory to save the model in
            filename:   filename for saved model
        """

        filename += '_ep_' + str(int(self.epoch-1)) + '.pt'

        if self.model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")

        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)

        dict1 = {
            'model_type': self.model_type,
            'model_hyperparams': self.model.model_hyperparams,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'detectors': self.detectors
        }

        if self.scheduler is not None:
            dict1['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(dict1, p / filename)

        if not os.path.exists(p/aux_filename):

            f = h5py.File(p / aux_filename, 'w')

            print('Params mean')
            print(self.training_wg.params_mean)
            print('Params std')
            print(self.training_wg.params_std)

            f.create_dataset('parameters_mean', data=self.training_wg.params_mean)
            f.create_dataset('parameters_std', data=self.training_wg.params_std)

            if self.training_wg.add_glitch and not self.training_wg.extrinsic_at_train:
                f.create_dataset('glitch_parameters_mean', data=self.training_wg.glitch_params_mean)
                f.create_dataset('glitch_parameters_std', data=self.training_wg.glitch_params_std)

            f.create_dataset('waveforms_mean', data=self.training_wg.means)
            f.create_dataset('waveforms_std', data=self.training_wg.stds)
            f.create_dataset('Vh_real', data=self.training_wg.svd.Vh.real)
            f.create_dataset('Vh_imag', data=self.training_wg.svd.Vh.imag)

            f.close()

        if not os.path.exists(p / 'hyperparams.txt'):

            f = open(p / 'hyperparams.txt', 'w')

            for key, value in self.model.model_hyperparams.items():
                if type(value) == dict:
                    f.write(key)
                    for k, v in value.items():
                        f.write(k+'\t'+str(v)+'\n')
                else:
                    f.write(key+'\t'+str(value)+'\n')

            f.write('learning rate'+'\t'+'%s\n'% (self.lr))
            f.write('batch size'+'\t'+'%d\n'% (self.batch_size))
            f.write('annealing method    ' + self.annealing)
            f.close()



    def load_model(self, filename='model'):
        """Load a saved model.
        Args:
            filename:       File name
        """

        if self.model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")

        p = Path(self.model_dir)

        if self.epoch_to_use != -1:
            # load a specific epoch
            filename += '_ep_' + str(self.epoch_to_use) + '.pt'
            if os.path.exists(p/filename):
                checkpoint = torch.load(p / filename, map_location=self.device)
            else:
                print('Invalid epoch specified')
                return
        else:
            # load the latest epoch
            # list files
            model_epoch = [int((f[:-3]).split('_')[-1]) for f in os.listdir(p) if f.startswith(filename)]
            self.epoch_to_use = np.max(model_epoch)
            filename += '_ep_' + str(self.epoch_to_use) + '.pt'

            checkpoint = torch.load(p / filename, map_location=self.device)

        model_type = checkpoint['model_type']
        model_hyperparams = checkpoint['model_hyperparams']


        # Load model
        self.construct_model(model_type, existing=True, **model_hyperparams)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Load optimizer
        scheduler_present_in_checkpoint = ('scheduler_state_dict' in
                                           checkpoint.keys())


        # If the optimizer has more than 1 param_group, then we built it with
        # flow_lr different from lr
        if len(checkpoint['optimizer_state_dict']['param_groups']) > 1:
            flow_lr = (checkpoint['optimizer_state_dict']['param_groups'][-1]
                       ['initial_lr'])
        else:
            flow_lr = None

        self.initialize_training(lr_annealing=scheduler_present_in_checkpoint,
                                 flow_lr=flow_lr)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        if scheduler_present_in_checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Set the epoch to the correct value. This is needed to resume
        # training.
        self.epoch = self.epoch_to_use+1


        # Store the list of detectors the model was trained with
        self.detectors = checkpoint['detectors']

        # Load history
        i = 1
        with open(p / 'history.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if i == self.epoch:
                    break
                self.train_history.append(float(row[1]))
                self.test_history.append(float(row[2]))
                i += 1

        # Make sure the model is in evaluation mode
        self.model.eval()



    def plot_losses(self):

        epoch_axis = np.arange(1, self.epoch)

        fig, ax = plt.subplots()
        fig.set_size_inches(7, 7)

        ax.plot(epoch_axis, self.train_history, label='Training Loss')
        ax.plot(epoch_axis, self.test_history, label='Validation Loss')

        legend = ax.legend()

        fig.savefig(self.model_dir+'losses.png')




    def train(self, epochs, output_freq=50, kl_annealing=True,
              snr_annealing=False, save_once_in=None):
        """Train the model.
        Args:
                epochs:     number of epochs to train for
                output_freq:    how many iterations between outputs
                kl_annealing:  for cvae, whether to anneal the kl loss
        """

        # if self.wfd.extrinsic_at_train:
        #     add_noise = False
        # else:
        #     add_noise = True
        add_noise = True

        start = self.epoch
        end = self.epoch + epochs

        for epoch in range(start, end):

            print('Learning rate: {:e}'.format(self.optimizer.state_dict()
                                               ['param_groups'][0]['lr']))

            if self.model_type == 'nde':
                train_loss = nde_flows.train_epoch(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    epoch,
                    self.device,
                    output_freq,
                    add_noise,
                    snr_annealing)

                test_loss = nde_flows.test_epoch(
                    self.model,
                    self.test_loader,
                    epoch,
                    self.device,
                    add_noise,
                    snr_annealing)


            if self.scheduler is not None:
                self.scheduler.step()

            self.epoch += 1
            self.train_history.append(train_loss)
            self.test_history.append(test_loss)

            # Log the history to file
            if self.model_dir is not None:
                p = Path(self.model_dir)
                p.mkdir(parents=True, exist_ok=True)

                # Make column headers if this is the first epoch
                if epoch == 1:
                    with open(p / 'history.txt', 'w') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([epoch, train_loss, test_loss])

                else:
                    with open(p / 'history.txt', 'a') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([epoch, train_loss, test_loss])

            if save_once_in is not None and (epoch+1-start) % save_once_in == 0:
                print('Saving model\n')
                self.save_model()

        # save the model at the end
        print('Saving model')
        self.save_model()


    def init_waveform_supp(self, aux_filename='waveforms_supplementary.hdf5', compute_fisher=False,
                           compute_bilby_post=False):

        p = Path(self.model_dir)

        try:
            f = h5py.File(p / aux_filename, 'r')
        except FileNotFoundError:
            return

        self.testing_wg = wd.WaveformGenerator(directory=self.data_dir)

        if self.test_on_training_data:
            self.testing_wg.load_data('training_data')
        else:
            self.testing_wg.load_data('testing_data')

        self.testing_wg.params_mean = f['parameters_mean'][:]
        self.testing_wg.params_std = f['parameters_std'][:]

        if self.testing_wg.add_glitch and not self.testing_wg.extrinsic_at_train:
            self.testing_wg.glitch_params_mean = f['glitch_parameters_mean'][:]
            self.testing_wg.glitch_params_std = f['glitch_parameters_std'][:]

        if not self.testing_wg.extrinsic_at_train:
            self.testing_wg.means = f['waveforms_mean'][:]
            self.testing_wg.stds = f['waveforms_std'][:]

        Vh = f['Vh_real'][:] + 1j*f['Vh_imag'][:]

        if not self.test_on_training_data:
            # do the svd
            self.testing_wg.perform_svd(Vh)

            #print(self.testing_wg.params_mean)
            #print(self.testing_wg.params_std)

            if self.testing_wg.extrinsic_at_train:
                self.testing_wg.normalize_params()
                #self.testing_wg.normalize_dataset()

        f.close()

        if compute_fisher:
            self.fisher = Fisher(waveform_generator=self.testing_wg)
        if compute_bilby_post:
            self.bilbly_post = Bilby_Posterior(self.testing_wg, self.model_dir)


    def evaluate(self, idx, nsamples=10000, plot=True, compute_fisher=False, compute_bilby_post=False):
        """Evaluate the model on a noisy waveform.
        Args:
            idx         index of the waveform, from a noisy waveform
                        database
            plot        whether to make a corner plot
        """

        idx = idx // self.testing_wg.noise_real_to_sig
        det = -100


        if self.testing_wg.add_glitch:
            y, params_true, det = self.testing_wg.provide_sample(idx, return_det=True)
        else:
            y, params_true = self.testing_wg.provide_sample(idx, return_det=False)

        if self.model_type == 'nde':
            x_samples = nde_flows.obtain_samples(
                self.model, y, nsamples, self.device
            )

        x_samples = x_samples.to(self.device)

        params_samples = self.testing_wg.post_process_parameters(x_samples.cpu().numpy())

        # find maximum likelihood point
        params_samples_ml = np.zeros(params_samples.shape[1])
        for i in range(0, params_samples.shape[1]):

            bins, edges = np.histogram(params_samples[:,i], bins=20)
            bins = scipy.ndimage.gaussian_filter(bins, sigma=0.5)
            params_samples_ml[i] = (edges[np.argmax(bins)]+edges[np.argmax(bins)+1])/2.

        params_true = self.testing_wg.post_process_parameters(params_true)
        print('True parameters: ', params_true)
        print('ML estimations: ', params_samples_ml)

        if compute_fisher:

            cov_matrix = self.fisher.compute_analytical_cov_m1_m2(params=params_true)
            print('Covariance matrix:')
            print(cov_matrix)

        slice = [0, 1, 2]

        #limit_low = np.percentile(params_samples[:,slice], 1, axis=0)
        #limit_high = np.percentile(params_samples[:, slice], 99, axis=0)
        limit_low = np.asarray([25., 20., 100.])
        limit_high = np.asarray([60., 50., 1100.])
        range_not_zoomed = np.stack((limit_low, limit_high), axis=1)
        limit_low = params_samples_ml*0.85
        limit_high = params_samples_ml*1.15
        range_zoomed = np.stack((limit_low, limit_high), axis=1)

        print(range_zoomed)

        if plot:

            fig1 = corner.corner(params_samples[:, slice], truths=params_true[slice],
                          labels=parameter_labels[slice], hist_kwargs={"density":True}, bins=20)

            # plt.show()
            plt.savefig(self.model_dir+str(idx))

            fig_not_zoomed = corner.corner(params_samples[:, slice], truths=params_true[slice],
                                           labels=parameter_labels[slice], hist_kwargs={"density":True},
                                           bins=20, range=range_not_zoomed,plot_datapoints=False,
                                           no_fill_contours=False, fill_contours=True,
                                           levels=(0.3935, 0.8647, 0.9889, 0.9997))

            fig_zoomed = corner.corner(params_samples[:, slice], truths=params_true[slice],
                                           labels=parameter_labels[slice], hist_kwargs={"density":True},
                                           bins=20, range=range_zoomed,plot_datapoints=False,
                                           no_fill_contours=False, fill_contours=True,
                                           levels=(0.3935, 0.8647, 0.9889, 0.9997))

            if compute_fisher:

                plot_fisher_estimates(fig_not_zoomed, range_not_zoomed, params_samples_ml,
                                      cov_matrix)

                plot_fisher_estimates(fig_zoomed, range_zoomed, params_samples_ml,
                                      cov_matrix)

            if compute_bilby_post:

                print('Computing bilby posteriors...')
                bilby_result, _ = self.bilbly_post.find_result(idx, params_true)
                print('Posteriors computed')

                print(bilby_result.samples)

                # corner.corner(bilby_result.samples, labels=parameter_labels[slice], hist_kwargs={"density": True},
                #                                bins=20, range=range_not_zoomed, plot_datapoints=False,
                #                                no_fill_contours=False, fill_contours=True,
                #                                levels=(0.3935, 0.8647, 0.9889, 0.9997), fig=fig_not_zoomed)
                corner.corner(bilby_result.samples, labels=parameter_labels[slice], hist_kwargs={"density": True},
                              bins=20, range=range_not_zoomed, plot_datapoints=False,
                              no_fill_contours=False, fill_contours=True,
                              levels=(0.3935, 0.8647, 0.9889, 0.9997), fig=fig_not_zoomed,
                              color='m')

                corner.corner(bilby_result.samples, labels=parameter_labels[slice], hist_kwargs={"density": True},
                                           bins=20, range=range_zoomed, plot_datapoints=False,
                                           no_fill_contours=False, fill_contours=True,
                                           levels=(0.3935, 0.8647, 0.9889, 0.9997), fig=fig_zoomed,
                                           color='m')


            fig_not_zoomed.savefig(self.model_dir + str(idx))
            fig_zoomed.savefig(self.model_dir + str(idx)+'_zoomed')

            # if self.testing_wg.add_glitch:
            #
            #     slice = np.arange(3, 15)
            #
            #     corner.corner(params_samples[:, slice], truths=params_true[slice],
            #                   labels=parameter_labels[slice], hist_kwargs={"density": True},
            #                   bins=20)
            #         # plt.show()
            #     plt.savefig(self.model_dir + str(idx) + 'glitch')

        return params_samples


def print_gpu_info(num_gpus):

    num_available_gpus = torch.cuda.device_count()
    print('# of available GPUs: ', str(num_available_gpus))

    for i in range(0, num_available_gpus):
        print(torch.cuda.get_device_name(i))
        print(torch.cuda.get_device_properties(i))

    if num_gpus == 1:

        print('Using ' + torch.cuda.get_device_name(0))

    else:
        print('**Parallel computing**')

        if num_gpus > num_available_gpus:
            print('Desired number of GPUs is larger than the number of available GPUs.')
            num_gpus = num_available_gpus

        print('Using: \n')
        for i in range(0, num_gpus):
            print(torch.cuda.get_device_name(0))

        print('*******************')


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

        axes[4 * k].plot(x, norm.pdf(x, loc=params_samples_ml[k],
                                    scale=np.sqrt(cov_matrix[k, k])), 'r-')

        for l in range(k + 1, 3):
            plot_gauss_contours(params_samples_ml, cov_matrix, k, l,
                                    axes[3 * l + k])


class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Model the gravitational-wave parameter '
                     'posterior distribution with neural networks.'))

    # Since options are often combined, defined parent parsers here and pass
    # them as parents when defining ArgumentParsers.

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--data_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false',
                                   dest='cuda')
    dir_parent_parser.add_argument('--gpus', type=int, default='1',
                                   dest='num_gpus')

    activation_parent_parser = argparse.ArgumentParser(add_help=None)
    activation_parent_parser.add_argument(
        '--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu')

    train_parent_parser = argparse.ArgumentParser(add_help=None)
    train_parent_parser.add_argument(
        '--batch_size', type=int, default='512')
    train_parent_parser.add_argument('--lr', type=float, default='0.0001')
    train_parent_parser.add_argument('--lr_anneal_method',
                                     choices=['step', 'cosine', 'cosineWR'],
                                     default='step')
    train_parent_parser.add_argument('--no_lr_annealing', action='store_false',
                                     dest='lr_annealing')
    train_parent_parser.add_argument(
        '--steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument('--steplr_step_size', type=int,
                                     default=80)
    train_parent_parser.add_argument('--flow_lr', type=float)
    train_parent_parser.add_argument('--epochs', type=int, required=True)
    train_parent_parser.add_argument(
        '--output_freq', type=int, default='50')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')
    train_parent_parser.add_argument('--save_once_in', type=int, dest='save_once_in',default=-1)
    train_parent_parser.add_argument('--no_kl_annealing', action='store_false',
                                     dest='kl_annealing')
    train_parent_parser.add_argument('--detectors', nargs='+')
    train_parent_parser.add_argument('--truncate_basis', type=int)
    train_parent_parser.add_argument('--snr_threshold', type=float)
    train_parent_parser.add_argument('--distance_prior_fn',
                                     choices=['uniform_distance',
                                              'inverse_distance',
                                              'linear_distance',
                                              'inverse_square_distance',
                                              'bayeswave'])
    train_parent_parser.add_argument('--snr_annealing', action='store_true')
    train_parent_parser.add_argument('--distance_prior', type=float,
                                     nargs=2)
    train_parent_parser.add_argument('--bw_dstar', type=float)


    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    train_parser = mode_subparsers.add_parser('train', description=('Train a network.'))

    test_parser = mode_subparsers.add_parser('test', description=('Test a network trained beforehand.'), parents=[
                 dir_parent_parser])

    test_parser.add_argument('--epoch', dest='epoch_to_use', default=-1, type=int)
    test_parser.add_argument('--test_on_training_data', dest='test_on_training_data', action='store_true')
    test_parser.add_argument('--fisher', dest='compute_fisher', action='store_true')
    test_parser.add_argument('--bilby', dest='compute_bilby_post', action='store_true')

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'))

    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = True


    # nde (curently just NSFC)

    nde_parser = type_subparsers.add_parser(
        'nde',
        description=('Build and train a flow from the nde package.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser]
    )
    nde_parser.add_argument('--hidden_dims', type=int, required=True)
    nde_parser.add_argument('--nflows', type=int, required=True)
    nde_parser.add_argument('--batch_norm', action='store_true')
    nde_parser.add_argument('--nbins', type=int, required=True)
    nde_parser.add_argument('--tail_bound', type=float, default=1.0)
    nde_parser.add_argument('--apply_unconditional_transform',
                            action='store_true')
    nde_parser.add_argument('--dropout_probability', type=float, default=0.0)
    nde_parser.add_argument('--num_transform_blocks', type=int, default=2)
    nde_parser.add_argument('--base_transform_type', type=str,
                            choices=['rq-coupling', 'rq-autoregressive'],
                            default='rq-coupling')

    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, train_parent_parser])

    ns = Nestedspace()

    return parser.parse_args(namespace=ns)


def main():

    args = parse_args()

    print('Waveform directory', args.data_dir)
    print('Model directory', args.model_dir)
    pm = PosteriorModel(model_dir=args.model_dir,
                        data_dir=args.data_dir,
                        use_cuda=args.cuda)
    print('Device', pm.device)

    if pm.device.type == 'cuda':

        num_gpus = args.num_gpus
        print_gpu_info(num_gpus)

    if args.mode == 'train':

        print('Loading dataset')
        print('Batch size: ', str(args.batch_size))
        pm.load_dataset(batch_size=args.batch_size)
        pm.batch_size = args.batch_size

        if args.model_source == 'new':

            print('\nConstructing model of type', args.model_type)

            if args.model_type == 'nde':
                pm.construct_model(
                    'nde',
                    num_flow_steps=args.nflows,
                    base_transform_kwargs={
                        'hidden_dim': args.hidden_dims,
                        'num_transform_blocks': args.num_transform_blocks,
                        'activation': args.activation,
                        'dropout_probability': args.dropout_probability,
                        'batch_norm': args.batch_norm,
                        'num_bins': args.nbins,
                        'tail_bound': args.tail_bound,
                        'apply_unconditional_transform': args.apply_unconditional_transform,
                        'base_transform_type': args.base_transform_type
                    }
                )
            else:
                print('Wrong model type')

            print('\nInitial learning rate', args.lr)
            if args.lr_annealing is True:
                pm.annealing = args.lr_anneal_method
                if args.lr_anneal_method == 'step':
                    print('Stepping learning rate by', args.steplr_gamma,
                          'every', args.steplr_step_size, 'epochs')
                elif args.lr_anneal_method == 'cosine':
                    print('Using cosine LR annealing.')
                elif args.lr_anneal_method == 'cosineWR':
                    print('Using cosine LR annealing with warm restarts.')
            else:
                print('Using constant learning rate. No annealing.')
            if args.flow_lr is not None:
                print('Autoregressive flows initial lr', args.flow_lr)
            pm.initialize_training(lr=args.lr,
                                   lr_annealing=args.lr_annealing,
                                   anneal_method=args.lr_anneal_method,
                                   total_epochs=args.epochs,
                                   # steplr=args.steplr,
                                   steplr_step_size=args.steplr_step_size,
                                   steplr_gamma=args.steplr_gamma,
                                   flow_lr=args.flow_lr)

        elif args.model_source == 'existing':

            pm.epoch_to_use = -1

            print('Loading existing model')
            pm.load_model()

        print('\nModel hyperparameters:')
        for key, value in pm.model.model_hyperparams.items():
            if type(value) == dict:
                print(key)
                for k, v in value.items():
                    print('\t', k, '\t', v)
            else:
                print(key, '\t', value)

        print('\nTraining for {} epochs'.format(args.epochs))

        print('Starting timer')
        start_time = time.time()

        if args.save_once_in != -1:
            save_once_in = args.save_once_in
        else:
            # save at the very end
            save_once_in = None

        if args.save:
            pm.train(args.epochs,
                     output_freq=args.output_freq,
                     kl_annealing=args.kl_annealing,
                     snr_annealing=args.snr_annealing,
                     save_once_in=save_once_in)

        else:
            pm.train(args.epochs,
                     output_freq=args.output_freq,
                     kl_annealing=args.kl_annealing,
                     snr_annealing=args.snr_annealing)

        print('Stopping timer.')
        stop_time = time.time()
        print('Training time (including validation): {} seconds'
              .format(stop_time - start_time))


    elif args.mode == 'test':

        pm.test_on_training_data = args.test_on_training_data
        print('Test on training data is ', pm.test_on_training_data)
        pm.epoch_to_use = args.epoch_to_use

        print('Loading existing model')
        pm.load_model()
        pm.plot_losses()

        print('\nModel hyperparameters:')

        for key, value in pm.model.model_hyperparams.items():
            if type(value) == dict:
                print(key)
                for k, v in value.items():
                    print('\t', k, '\t', v)
            else:
                print(key, '\t', value)

        # TESTING
        print('Testing is starting...')
        pm.init_waveform_supp(compute_fisher=args.compute_fisher, compute_bilby_post=args.compute_bilby_post)

        for i in range(0, 10):

            idx = np.random.randint(0, (pm.testing_wg.dataset_len*pm.testing_wg.noise_real_to_sig))
            #idx = 242
            # print(idx)
            pm.evaluate(idx, plot=True, compute_fisher=args.compute_fisher, compute_bilby_post=args.compute_bilby_post)

    else:
        print('Wrong mode selected')

    print('Program complete')


if __name__ == "__main__":
    main()
