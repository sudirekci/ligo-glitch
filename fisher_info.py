import numpy as np
import matplotlib.pyplot as plt


class Fisher:

    step_sizes = dict(mass1=1e-4, mass2=1e-4, distance=1e-4)
    sm_to_sec = 4.926*10**(-6)

    def __init__(self, waveform_generator=None, params=None, elements=None):

        if elements is None:
            elements = ['mass1', 'mass2', 'distance']

        self._wg = waveform_generator
        # all the true parameters of the waveform as a dict
        self._params = params
        # parameters with respect to which the Fisher info will be computed
        self._elements = elements

        self.F = None
        self.F_inv = None
        self.derivatives = {key: None for key in elements}
        self.th_rms = None
        self.calc_rms = None

        self.psd = np.zeros(len(self._wg.freqs))
        self.psd[self._wg.freqs<10] = np.inf
        f0 = 70
        self.psd = 3*10**(-48)*((f0/self._wg.freqs)**4+2*(1+self._wg.freqs**2/f0**2))


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

                    #self.F[m, i, j] = self._wg.inner_whitened(self.derivatives[self._elements[i]][m,:],
                    #                                 self.derivatives[self._elements[j]][m,:])
                self.F[i, j] = inner

        #for m in range(0, self._wg.no_detectors):
        self.F = self.F + self.F.T - np.diag(self.F.diagonal())

        #print('Analytical (m1 m2) Fisher Matrix')
        #print(self.F)

        return self.F


    def compute_analytical_cov_m1_m2(self, index=-1):

        if self._params is None and index == -1:
            print('Set parameters first')
            return -1

        self.compute_fisher_matrix(index=index)

        # self.F_inv = 1./np.sum(1./np.linalg.inv(self.F), axis=0)
        self.F_inv = np.linalg.inv(self.F)

        #print('Calculated Fisher')
        #print(self.F)

        return self.F_inv

    def compute_theoretical_cov_mu_chirp(self, index):

        self._params = np.copy(self._wg.params[index, :])

        # th[0] = ΔM1/M1, th[1] = ΔM2/M2, th[2] = ΔD/D
        self.th_rms = np.zeros(3)

        self.th_rms[2] = 1./np.sqrt(self._wg.snrs[0, index]**2+self._wg.snrs[1, index]**2)
        # self.th_rms[2] = 1./np.sqrt(self._wg.snrs[0, index]**2)

        hp, hc = self._wg.compute_hp_hc(index)
        self._wg.project_hp_hc(hp, hc, index, whiten=False)

        h = self._wg.projection_strains.copy()

        m1 = self._params[self._wg.INTRINSIC_PARAMS['mass1']]*self.sm_to_sec
        m2 = self._params[self._wg.INTRINSIC_PARAMS['mass2']]*self.sm_to_sec

        M = (m1+m2)
        mu = m1*m2/M
        chirp_mass = (m1*m2)**(3/5)/M**(1/5)

        # print('M1', m1/self.sm_to_sec)
        # print('M2', m2 / self.sm_to_sec)
        # print('MU', mu / self.sm_to_sec)
        # print('CHIRP MASS', chirp_mass / self.sm_to_sec)
        #
        # print("SNR", np.sqrt(self._wg.snrs[0, index] ** 2 + self._wg.snrs[1, index] ** 2))
        # print("1/SNR", 1./np.sqrt(self._wg.snrs[0,index]**2+self._wg.snrs[1,index]**2))

        f = self._wg.freqs[self._wg.fft_mask]

        x = (np.pi*M*f)**(2/3)

        derivs = np.zeros((3, self._wg.no_detectors, len(self._wg.projection_strains[0])),
                                                         dtype=np.complex64)
        f0 = 10

        for i in range(0, self._wg.no_detectors):

            derivs[0, i, :] = 1j*(0.75*np.power(8*np.pi*chirp_mass*f,-5./3)*h[i]*
                             ((-3715./756+55*mu/(6*M))*x+24*np.pi*np.power(x, 3./2)))
            derivs[1, i, :] = -1j*(5/4*np.power(8*np.pi*chirp_mass*f,-5./3)*h[i]*
                              (1.+55*mu/(6*M)*x+8*np.pi*np.power(x, 3./2)))
            derivs[2, i, :] = h[i]

        th_fisher = np.zeros((3, 3))

        for i in range(0, 3):
            for j in range(0, 3):

                inner = 0.
                for m in range(0, self._wg.no_detectors):

                    inner += self._wg.inner_colored(derivs[i, m, :], derivs[j, m, :])

                    # inner += self._wg.inner_colored(derivs[i, m, :],
                    #                               derivs[j, m+1//self._wg.no_detectors, :])

                th_fisher[i, j] = inner


        th_cov = np.linalg.inv(th_fisher)

        #print('MU RMS')
        #print(np.sqrt(1./(1./rms_th[0, 0, 0]+1./rms_th[1, 0, 0])))
        #print(np.sqrt(rms_th[0, 0]))
        #print('CHIRP RMS %')
        #print(np.sqrt(1./(1./rms_th[0, 1, 1]+1./rms_th[1, 1, 1])))
        #print(np.sqrt(rms_th[1, 1])*100)

        mu_rms = np.sqrt(th_cov[0, 0])

        self.th_rms[0] = mu_rms*np.abs((M*(mu-3*m1))/(2*mu*(m1-m2)))*mu/m1
        self.th_rms[1] = mu_rms*np.abs((M*(mu-3*m2))/(2*mu*(m1-m2)))*mu/m2

        return th_fisher, th_cov

    def compute_theoretical_cov_m1_m2(self, index):

        th_fisher, _ = self.compute_theoretical_cov_mu_chirp(index)

        print(th_fisher)

        m1 = self._params[self._wg.INTRINSIC_PARAMS['mass1']]
        m2 = self._params[self._wg.INTRINSIC_PARAMS['mass2']]
        distance = self._params[self._wg.EXTRINSIC_PARAMS['distance']]

        J = np.asarray([[m2/(m1**2+m1*m2), m1/(m2**2+m1*m2),0],
                        [(2*m1+3*m2)/(5*m1**2+5*m1*m2), (3*m1+2*m2)/(5*m2**2+5*m1*m2),0],
                        [(2*m1+3*m2)/(6*m1**2+6*m1*m2),(2*m2+3*m1)/(6*m2**2+6*m1*m2),-1/distance]])

        m1_m2_fisher = np.dot(np.dot(J.T, th_fisher), J)
        th_cov = np.linalg.inv(m1_m2_fisher)

        return th_cov


    def compute_calc_rms(self):

        # assume that Fisher matrix etc. are already calculated
        self.calc_rms = np.zeros(3)

        self.calc_rms[0] = np.sqrt(self.F_inv[0, 0])/self._params[self._wg.INTRINSIC_PARAMS['mass1']]
        self.calc_rms[1] = np.sqrt(self.F_inv[1, 1]) / self._params[self._wg.INTRINSIC_PARAMS['mass2']]
        self.calc_rms[2] = np.sqrt(self.F_inv[2, 2]) / self._params[self._wg.EXTRINSIC_PARAMS['distance']]


    def compute_analytical_cov_mu_chirp(self, index):

        self._params = np.copy(self._wg.params[index, :])

        # th[0] = ΔM1/M1, th[1] = ΔM2/M2, th[2] = ΔD/D
        self.analy_rms = np.zeros(3)

        self.analy_rms[2] = 1./np.sqrt(self._wg.snrs[0, index]**2+self._wg.snrs[1, index]**2)

        m1 = self._params[self._wg.INTRINSIC_PARAMS['mass1']]
        m2 = self._params[self._wg.INTRINSIC_PARAMS['mass2']]
        distance = self._params[self._wg.EXTRINSIC_PARAMS['distance']]

        M = (m1+m2)
        mu = m1*m2/M
        chirp_mass = (m1*m2)**(3/5)/M**(1/5)

        step_size = 1e-4

        derivs2 = np.zeros((3, self._wg.no_detectors, len(self._wg.projection_strains[0])),
                                        dtype=np.complex64)

        chirp_low = chirp_mass - step_size
        chirp_high = chirp_mass + step_size

        mu_low2 = mu - 2*step_size
        mu_low = mu - 1e-4
        mu_high = mu + 1e-4
        mu_high2 = mu + 2*step_size

        dist_low = distance-step_size
        dist_high = distance+step_size

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, dist_low])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, dist_low])

        f_low = self._wg.projection_strains.copy()

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, dist_high])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, dist_high])

        f_high = self._wg.projection_strains.copy()

        for j in range(0, self._wg.no_detectors):

            derivs2[2, j, :] = (f_high[j]-f_low[j])/(2*step_size)*distance

        m1, m2 = self.from_mu_chirp_to_m1_m2(mu_low, chirp_mass)

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, distance])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, distance])

        f_low = self._wg.projection_strains.copy()

        m1, m2 = self.from_mu_chirp_to_m1_m2(mu_high, chirp_mass)

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, distance])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, distance])

        f_high = self._wg.projection_strains.copy()

        for j in range(0, self._wg.no_detectors):

            derivs2[0, j, :] = (f_high[j]-f_low[j])/(2*step_size)*mu


        m1, m2 = self.from_mu_chirp_to_m1_m2(mu, chirp_low)

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, distance])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, distance])

        f_low = self._wg.projection_strains.copy()

        m1, m2 = self.from_mu_chirp_to_m1_m2(mu, chirp_high)

        hp, hc = self._wg.compute_hp_hc(-1, params=[m1, m2, distance])
        self._wg.project_hp_hc(hp, hc, -1, params=[m1, m2, distance])

        f_high = self._wg.projection_strains.copy()

        for j in range(0, self._wg.no_detectors):
            derivs2[1, j, :] = (f_high[j] - f_low[j]) / (2 * step_size)*chirp_mass


        analy_fisher = np.zeros((3, 3))

        for i in range(0, 3):
            for j in range(0, i+1):

                inner = 0.
                for m in range(0, self._wg.no_detectors):

                    inner += self._wg.inner_whitened(derivs2[i, m, :], derivs2[j, m, :])

                    # inner += self._wg.inner_colored(derivs[i, m, :],
                    #                               derivs[j, m+1//self._wg.no_detectors, :])

                analy_fisher[i, j] = inner

        analy_fisher = analy_fisher + analy_fisher.T - np.diag(analy_fisher.diagonal())

        analy_cov = np.linalg.inv(analy_fisher)

        return analy_fisher,analy_cov

    def compute_analytical_cov_m1_m2_from_mu_chirp(self, index):

        analy_fisher,_ = self.compute_analytical_cov_mu_chirp(index)

        m1 = self._params[self._wg.INTRINSIC_PARAMS['mass1']]
        m2 = self._params[self._wg.INTRINSIC_PARAMS['mass2']]
        distance = self._params[self._wg.EXTRINSIC_PARAMS['distance']]

        J = np.asarray([[m2/(m1**2+m1*m2), m1/(m2**2+m1*m2),0],
                        [(2*m1+3*m2)/(5*m1**2+5*m1*m2), (3*m1+2*m2)/(5*m2**2+5*m1*m2),0],
                        [(2*m1+3*m2)/(6*m1**2+6*m1*m2),(2*m2+3*m1)/(6*m2**2+6*m1*m2),-1/distance]])

        m1_m2_fisher = np.dot(np.dot(J.T, analy_fisher), J)
        analy_cov = np.linalg.inv(m1_m2_fisher)

        return analy_cov



    def from_mu_chirp_to_m1_m2(self,mu, chirp):

        m1 = 1/2*(chirp**(5/2)*mu**(-3/2)+
                     np.sqrt(chirp**5*mu**(-3)-
                     4*chirp**(5/2)*mu**(-1/2)))

        m2 = 1/2*(chirp**(5/2) * mu ** (-3 / 2)-
                     np.sqrt(chirp ** 5 * mu ** (-3) -
                     4 * chirp ** (5 / 2) * mu ** (-1 / 2)))

        return (m1, m2)

