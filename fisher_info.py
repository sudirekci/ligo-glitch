import numpy as np
import matplotlib.pyplot as plt


class Fisher:

    step_sizes = dict(mass1=1e-3, mass2=1e-3, distance=1e-1)
    #step_sizes = dict(mass1=1e-6, mass2=1e-6, distance=1e-3)


    def __init__(self, waveform_generator=None, params=None, elements=None):

        if elements is None:
            elements = ['mass1', 'mass2', 'distance']

        self._wg = waveform_generator
        # all the true parameters of the waveform as a dict
        self._params = params
        # parameters with respect to which the Fisher info will be computed
        self._elements = elements

        self.F = None
        self.derivatives = {key: None for key in elements}


    @property
    def wg(self):
        return self._wg

    @property
    def params(self):
        return self._params

    @property
    def elements(self):
        return self._elements

    @wg.setter
    def wg(self, waveform_generator):
        self._wg = waveform_generator

    @params.setter
    def params(self, params_new):
        self._params = params_new

    @elements.setter
    def elements(self, elements_new):
        self._elements = elements_new


    def take_derivative(self, el):

        # for example, el = 'mass1'
        if el in self._wg.INTRINSIC_PARAMS:
            index = self._wg.INTRINSIC_PARAMS[el]
        else:
            index = self._wg.EXTRINSIC_PARAMS[el]

        step_size = Fisher.step_sizes[el]
        param = self._params[index]
        param_low = param - step_size
        param_high = param + step_size

        #print('Element ', el)
        #print('Step size ', step_size)
        #print(param)

        # compute f(x-Δx)
        self._params[index] = param_low
        #print(self._params)

        hp, hc = self._wg.compute_hp_hc(-1, params=self._params)
        self._wg.project_hp_hc(hp, hc, -1, params=self._params)

        f_low = self._wg.projection_strains.copy()

        # compute f(x+Δx)
        self._params[index] = param_high
        #print(self._params)

        hp, hc = self._wg.compute_hp_hc(-1, params=self._params)
        self._wg.project_hp_hc(hp, hc, -1, params=self._params)

        f_high = self._wg.projection_strains.copy()

        # return parameter to original value
        self._params[index] = param

        self.derivatives[el] = np.zeros((self._wg.no_detectors, len(self._wg.projection_strains[0])),
                                        dtype=np.complex64)

        for j in range(0, self._wg.no_detectors):

            self.derivatives[el][j, :] = (f_high[j]-f_low[j])/(2*step_size)



    def compute_fisher_matrix(self, index=-1):

        if index != -1:
            self._params = np.copy(self._wg.params[index, :])

        # Fisher info matrix 3x3
        # Fij = (hi, hj)
        N = len(self._elements)
        self.F = np.zeros((N, N))

        # take derivatives
        for el in self._elements:
            self.take_derivative(el)

        for i in range(0, N):
            for j in range(0, i+1):

                inner = 0.
                for m in range(0, self._wg.no_detectors):
                #for m in range(0, 1):

                    inner += self._wg.inner_whitened(self.derivatives[self._elements[i]][m,:],
                                                     self.derivatives[self._elements[j]][m,:])

                self.F[i, j] = inner

        self.F = self.F + self.F.T - np.diag(self.F.diagonal())


    def compute_fisher_cov(self, index=-1):

        if self._params is None and index == -1:
            print('Set parameters first')
            return -1

        self.compute_fisher_matrix(index=index)
        return np.linalg.inv(self.F)
        #return self.F

