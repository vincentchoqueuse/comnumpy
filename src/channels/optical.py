from .core import Channel
from analysers.logger import Power_Reporter
from core import Sequential
from analysers.logger import Power_Reporter
import numpy as np

class CD(Channel):
    """ Chromatic Dispersion Channel 
    """

    def __init__(self, z, D=17*(10**-6), lamb=1.55 * 10**-6, c= 3*(10**8), F_s=1, alpha_dB=None, direction=1, name="cd"):
        self.F_s = F_s  # sampling frequency
        self.z = z  #  [m]
        self.lamb = lamb  # [m]
        self.D = D
        self.c = c
        self.name = name
        self.alpha_dB = alpha_dB
        self.direction = direction
        self.name = name

        self.prepare()

    def prepare(self):
        # second-order dispersion coefficient
        # see equation 3 in https://www.hindawi.com/journals/jfs/2022/8316404/
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304
        beta2 = -self.D * self.lamb**2 / (2*np.pi*self.c)

        if self.alpha_dB: 
            alpha = (np.log(10)/10)*(self.alpha_dB)
            gain = np.exp(-(alpha/2)*self.z*self.direction)
        else:
            gain = 1

        self.gain = gain 
        self.beta2 = beta2

    def forward(self, x):
        NFFT = len(x)
        w = (2*np.pi*self.F_s)*np.fft.fftfreq(NFFT, d=1)
        H = np.exp(1j * (self.beta2/2) * self.z * (w**2) * self.direction)
        fftx = np.fft.fft(x)
        ffty = H * fftx
        y = self.gain * np.fft.ifft(ffty)
        return y

    def to_dict(self):
        param = (self.beta2/2) * self.z * self.direction
        return {"param": param, "gain": self.gain, "z": self.z}


class NonLinearity(Channel):

    """ NonLinearity
    Reference:

    * Physics-Based Deep Learning for Fiber-Optic Communication Systems
    * https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304
    """
    def __init__(self, z, gamma, alpha_dB=0, direction=1, name="nl"):
        self.z = z  # [m]
        self.gamma = gamma  # nonlinear parameter [1/(W*m)]
        self.alpha_dB = alpha_dB
        self.direction = direction
        self.name = name

        self.prepare()

    def prepare(self):
        if self.alpha_dB :
            alpha = (np.log(10)/10) * (self.alpha_dB)
            gain = np.exp(-(alpha/2)*self.z*self.direction)
            L_eff = self.direction*(1-np.exp(-alpha*self.z))/alpha
        else:
            gain = 1
            L_eff = self.direction*self.z

        self.gain = gain
        self.nl_param = self.gamma*L_eff

    def forward(self, x):
        y = x * self.gain * np.exp(1j*self.nl_param*(np.abs(x)**2))
        return y 

    def to_dict(self):
        return {"param": self.nl_param, "gain": self.gain, "z": self.z}
   

class EDFA(Channel):

    """ EDFA
    
    See: https://www.sciencedirect.com/topics/engineering/spontaneous-emission-factor
    """

    def __init__(self, alpha_dB=0.2*(10**-3), L_span=80*(10**3), direction=1, name = "EDFA"):
        self.alpha_dB = alpha_dB  # [dB/m] 
        self.L_span  = L_span   # segment length [m]    
        self.direction = direction
        self.name = name

        self.prepare()

    def prepare(self):
        alpha_dB_tot = self.direction*self.alpha_dB*self.L_span 
        G = 10**(alpha_dB_tot/10)
        self.gain = np.sqrt(G)

    def forward(self,x):
        y = self.gain*x
        return y

    def to_dict(self):
        return {"gain": self.gain}


class ASE(Channel):

    """ ASE
    
    .. math ::

        N_{ase} = 2n_{sp} (G-1)h \nu_s B

    * :math:`B` is the optical noise bandwidth,
    * :math:`G=e^{\alpha L_A}` 
    * :math:`\alpha` is the fiber loss coefficient given by

    .. math :: 

        \alpha = \frac{\alpha_{dB}}{10 \log_{10}(e)}
    
    * :math:`n_{sp}` is the spontaneous-emission factor. This factor is linked to the Noise figure as

    .. math :: 

        N_F = 2n_{sp}\frac{G-1}{G}

    See: https://www.sciencedirect.com/topics/engineering/spontaneous-emission-factor
    """

    def __init__(self, alpha_dB=0.2*(10**-3), L_span=80*(10**3), F_s=1, NF_dB=5, scaling=2, h=6.626*(10**-34), nu=1.946*(10**14), name="ASE"):
        self.alpha_dB = alpha_dB  # [dB/m] 
        self.L_span  = L_span   # segment length [m]    
        self.NF_dB = NF_dB  # noise_figure [dB]
        self.h = h 
        self.nu_s = nu
        self.F_s = F_s  #  sampling frequency
        self.scaling = scaling
        self.name = name 

        self.prepare()

    def prepare(self):
        # see table 1 in https://arxiv.org/pdf/2010.14258.pdf
        B = self.F_s
        alpha_dB_tot = self.alpha_dB*self.L_span 
        G = 10**(alpha_dB_tot/10)
        NF = 10**(self.NF_dB/10)
        n_sp = (NF/2)/(1-1/G) # sponteaneous emission see table1
        p_ase  =  self.scaling * n_sp * (G-1) * self.h * self.nu_s * B
        self.p_ase = p_ase
        
    def forward(self,x):
        N = len(x)
        noise = np.sqrt(self.p_ase/2) * (np.random.randn(N) + 1j * np.random.randn(N))
        y = x + noise 
        return y

    def to_dict(self):
        return {"power": self.p_ase}


class Fiber_Link(Sequential):

    """Fiber_Link
    
    A Multi_Span_Link link composed of N span

    see: Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems
    """

    direction = 1

    def __init__(self, N_span, StPS, L_span, gamma=1.3*1e-3, lamb=1.55 * 10**-6, c=3*(10**8), h = 6.626*10**-34, nu=1.946*(10**14), alpha_dB= 0.2*1e-3, F_s=1, NF_dB = 4, noise_scaling=2, step_type="linear", step_log_factor=0.4, include_edfa=True, name="SSFM span"):
        self.N_span = N_span  
        self.StPS = StPS
        self.L_span = L_span
        self.gamma = gamma 
        self.alpha_dB = alpha_dB
        self.c = c
        self.h = h 
        self.lamb = lamb
        self.nu_s = nu
        self.F_s = F_s 
        self.NF_dB = NF_dB
        self.noise_scaling = noise_scaling
        self.step_type = step_type
        self.step_log_factor = step_log_factor
        self.include_edfa = include_edfa
        self.name = name 
        self.prepare()
        
    def prepare(self):
        """
        store the list of module
        """
        self.module_list = self.get_module_list()

    def get_step_size(self):
        """
        return the step sizes
        """
        N, L = self.StPS, self.L_span
        z = (L/N)*np.ones(N)
        return z

    def get_cd_module(self, dz):
        """
        return a chromatic dispersion layer
        """
        return CD(dz, alpha_dB=None, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=self.direction)

    def get_non_linear_module(self, dz):
        """
        return a non-linear layer
        """
        return NonLinearity(dz, self.gamma, alpha_dB=None, direction = self.direction)

    def get_ase_module(self):
        """
        return a noise layer
        """
        return ASE(self.alpha_dB, L_span = self.L_span, h=self.h, nu = self.nu_s, F_s=self.F_s, NF_dB=self.NF_dB, scaling=self.noise_scaling)

    def get_step_module_list(self, dz):
        """
        return the module list for a single step  of length dz
        """
        nl = self.get_non_linear_module(dz)
        cd = self.get_cd_module(dz)
        return [nl, cd]

    def get_span_module_list(self):
        """
        return the module list for a single span
        """
        z = self.get_step_size()
        
        module_list = []
        for num_step in range(self.StPS):
            dz = z[num_step]
            step_module_list = self.get_step_module_list(dz)
            module_list.extend(step_module_list)

        # end of span: add EDFA and ASE
        if self.include_edfa:
            edfa = EDFA(self.alpha_dB, L_span = self.L_span, direction=self.direction)
            module_list.append(edfa)

        ase = self.get_ase_module()
        module_list.append(ase)
        return module_list

    def get_module_list(self):
        """
        return the module list for the entire link of N_span
        """
        module_list = []
        
        for index in range(self.N_span):
            span = Sequential(self.get_span_module_list(), name="span {}".format(index+1))
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


class Logarithmic_Step_Size_Mixin:

    """
    Logarithmically spaced step size
    
    reference: 
    * [1] O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," in Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    * [2] J. Zhang, X. Li and Z. Dong, "Digital Nonlinear Compensation Based on the Modified Logarithmic Step Size," in Journal of Lightwave Technology, vol. 31, no. 22, pp. 3546-3555, Nov.15, 2013, doi: 10.1109/JLT.2013.2285648.
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


class Scheme_I_Step_Mixin:
    """
    SSFM with Loss With Dispersion (Scheme Ia)  / EQ 10

    reference: J. Shao, X. Liang and S. Kumar, "Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems," in IEEE Photonics Journal, vol. 6, no. 4, pp. 1-15, Aug. 2014, Art no. 7200515, doi: 10.1109/JPHOT.2014.2340993.
    """

    def get_cd_module(self, dz):
        return CD(dz, alpha_dB=self.alpha_dB, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=self.direction)

    def get_non_linear_module(self, dz):
        return NonLinearity(dz, self.gamma, direction = self.direction)
    

class Scheme_II_Step_Mixin:
    """
    SSFM with Loss With Nonlinearity (Scheme Ia)  / EQ 10

    reference: J. Shao, X. Liang and S. Kumar, "Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems," in IEEE Photonics Journal, vol. 6, no. 4, pp. 1-15, Aug. 2014, Art no. 7200515, doi: 10.1109/JPHOT.2014.2340993.
    """

    def get_cd_module(self, dz):
        return CD(dz, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=self.direction)

    def get_non_linear_module(self, dz):
        return NonLinearity(dz, self.gamma, alpha_dB=self.alpha_dB, direction = self.direction)

class Symmetric_Step_Mixin:

    def get_step_module_list(self, dz):
        cd1 = self.get_cd_module(dz/2)
        nl = self.get_non_linear_module(dz)
        cd2 = self.get_cd_module(dz/2)
        return [cd1, nl, cd2]

