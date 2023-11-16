from core import Sequential
from .optical import Chromatic_Dispersion, Kerr_NonLinearity, EDFA, ASE

import numpy as np

class Linear_Step_Size_Mixin:
    """
    Logarithmically spaced step size
    
    See also:

    * [1] O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," in Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    """

    def get_step_size(self):
        """
        return uniformly spaced step sizes
        """
        N, L = self.StPS, self.L_span
        z = (L/N)*np.ones(N)
        return z


class Logarithmic_Step_Size_Mixin:
    """
    Logarithmically spaced step size
    
    See also:

    *[1] O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," in Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    """

    step_log_factor = 0.4

    def get_step_size(self):
        """
        return logarithmically spaced step sizes (see equation 5 in [1])
        """
        N, L_span = self.StPS, self.L_span
        alpha = (np.log(10)/10) * (self.alpha_dB)
        alpha_adj = self.step_log_factor*alpha
        delta = (1-np.exp(-alpha_adj*L_span))/N
        N_vect = 1 + np.arange(N)
        
        if self.direction == 1:
            N_vect = N_vect[::-1]

        n_vect = N-N_vect+1
        z = -(1/alpha_adj)*np.log((1-n_vect*delta)/(1-(n_vect-1)*delta))
        return z


class Asymmetric_Step_Mixin:

    def get_step_module_list(self, dz):
        """
        return the module list for a single step of length dz
        """
        nl = self.get_non_linear_module(dz)
        cd = self.get_linear_module(dz)
        return [nl, cd]


class Symmetric_Step_Mixin:

    def get_step_module_list(self, dz):
        """
        return the module list for a single step  of length dz
        """
        cd1 = self.get_linear_module(dz/2)
        nl = self.get_non_linear_module(dz)
        cd2 = self.get_linear_module(dz/2)
        return [cd1, nl, cd2]


class Base_Fiber_Link(Sequential):
    """
    A Multi_Span_Link link composed of N spans. Each span is composed of StPS step. Each step is composed of 
    linear (CD) and non-linear (Kerr Nonlinearity) sections.

    :param N_span: Number of spans
    :param StPS: Number of step by span
    :param lamb: wavelength [nm]
    :param L_span: Length of span [km]
    :param alpha_dB: fiber loss parameter [dB/km]
    :param gamma: kerr coefficient [rad/W/km]
    :param c: speed of light [m/s]
    :param h: Planck’s constant [Js]
    :param nu: carrier frequency [Hz]


    See Also:

    * [1] J. Shao, X. Liang and S. Kumar, "Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems," in IEEE Photonics Journal, vol. 6, no. 4, pp. 1-15, Aug. 2014, Art no. 7200515, doi: 10.1109/JPHOT.2014.2340993.
    """

    direction = 1

    def __init__(self, N_span, StPS, L_span, gamma=1.3*10**-3, lamb=1553*(10**-9), c=3*(10**8), h = 6.626*10**-34, nu=1.946*(10**14), alpha_dB= 0.2, F_s=1, NF_dB = 4, noise_scaling=2, name="SSFM span"):
        self.N_span = N_span  
        self.StPS = StPS
        self.L_span = L_span  # in km
        self.gamma = gamma # in rad/W/m
        self.alpha_dB = alpha_dB # in dB/m
        self.c = c # in m/s
        self.h = h # in Js
        self.lamb = lamb # in nm
        self.nu_s = nu # in Hz
        self.F_s = F_s  # in Hz
        self.NF_dB = NF_dB
        self.noise_scaling = noise_scaling
        self.name = name 
        self.prepare()
        
    def prepare(self):
        """
        store the list of module
        """
        self.module_list = self.get_module_list()
  
    def get_linear_module(self, dz):
        """
        return a linear layer
        """
        return Chromatic_Dispersion(dz, alpha_dB=self.alpha_dB, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=self.direction)

    def get_non_linear_module(self, dz):
        """
        return a non-linear layer
        """
        return self.Kerr_NonLinearity(dz, self.gamma, alpha_dB=0, direction = self.direction)

    def get_pre_span_module_list(self):
        """
        return the module list before a span
        """
        return []

    def get_post_span_module_list(self):
        """
        return the module list after a span
        """
        edfa = EDFA(self.alpha_dB, L_span = self.L_span, direction=self.direction)
        ase = ASE(self.alpha_dB, L_span = self.L_span, h=self.h, nu = self.nu_s, F_s=self.F_s, NF_dB=self.NF_dB, scaling=self.noise_scaling)
        module_list = [edfa, ase]
        return module_list

    def get_span_module_list(self):
        """
        return the module list for a single span
        """
        z = self.get_step_size()
        
        # start of span
        module_list = self.get_pre_span_module_list()

        for num_step in range(self.StPS):
            dz = z[num_step]
            step_module_list = self.get_step_module_list(dz)
            module_list.extend(step_module_list)

        # end of span
        post_module_list = self.get_post_span_module_list()
        module_list.extend(post_module_list)
        return module_list

    def get_module_list(self):
        """
        return the module list for the entire link of N_span
        """
        module_list = []
        
        for index in range(self.N_span):
            span_module_list = self.get_span_module_list()
            span = Sequential(span_module_list, name="span {}".format(index+1))
            module_list.append(span)

        return module_list

    def get_gain(self):
        """
        return the total gain of the fiber link
        """
        gain = 1
        for module in self.get_span_module_list():
            if hasattr(module, 'gain'):
                gain = module.gain * gain
        return gain

    def get_extra_dict(self): 
        return {"name": self.name, 
                "Nb_spans": self.N_span,
                "StPS": self.StPS,
                "gain": self.get_gain()}


